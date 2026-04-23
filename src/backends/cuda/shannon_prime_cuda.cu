// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// CUDA kernels for VHT2 KV cache compression.
// Each kernel processes one or more independent vectors of length head_dim.
// The design prioritizes correctness and clarity; optimization follows.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <float.h>

// ============================================================================
// VHT2 Butterfly Kernel (p=2 stage of the Vilenkin-Hartley transform)
// ============================================================================
//
// Staged in-place Hartley transform. At p=2 the stage is the Hadamard
// butterfly scaled by 1/√2 per stage — self-inverse after log2(N) stages.
// Non-power-of-2 dimensions dispatch to the general staged kernel in
// shannon_prime_sqfree.cu (kernel_vilenkin_inplace), which handles the
// sqfree-padded case with primes {2,3,5,7,11}.
//
// Each thread block processes one vector of length n.
// Uses shared memory for the butterfly passes.
//
// For head_dim=128: 7 passes (log2(128)), each a full barrier.
// This is bandwidth-bound, not compute-bound — fine for KV cache writes
// which happen once per token per layer.

__global__ void kernel_vht2_p2(float *data, int n) {
    extern __shared__ float smem[];

    int vec_idx = blockIdx.x;              // Which vector
    int tid     = threadIdx.x;             // Thread within vector
    float *vec  = data + (size_t)vec_idx * n;

    // Load into shared memory
    if (tid < n) {
        smem[tid] = vec[tid];
    }
    __syncthreads();

    // Butterfly passes — each stage is a p=2 Hartley kernel normalised by
    // 1/√2. After log2(N) stages the total scale is 1/√N, making the whole
    // transform orthonormal and self-inverse (kernel_vht2_p2 applied
    // twice returns the original vector, no /N).
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (int len = 1; len < n; len <<= 1) {
        if (tid < n) {
            int grp    = tid / (len << 1);   // Which butterfly group
            int pos    = tid % (len << 1);   // Position within group
            int base   = grp * (len << 1);

            if (pos < len) {
                float u = smem[base + pos];
                float v = smem[base + pos + len];
                smem[base + pos]       = (u + v) * inv_sqrt2;
                smem[base + pos + len] = (u - v) * inv_sqrt2;
            }
        }
        __syncthreads();
    }

    // Write back
    if (tid < n) {
        vec[tid] = smem[tid];
    }
}

// ============================================================================
// Möbius Permutation Kernel
// ============================================================================
//
// Applies the squarefree-first reordering to VHT2 coefficients.
// order[i] = source index for position i in reordered vector.
// Each thread handles one element.

__global__ void kernel_mobius_reorder(const float *input, float *output,
                                     const int *order, int n) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        size_t out_off = (size_t)vec_idx * n + tid;
        size_t in_off  = (size_t)vec_idx * n + order[tid];
        output[out_off] = input[in_off];
    }
}

// Inverse: order[i] says "original index i goes to position order_inv[i]"
// We build the inverse table on host; kernel just gathers.
__global__ void kernel_mobius_unreorder(const float *input, float *output,
                                       const int *order, int n) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;

    if (tid < n) {
        // order[i] = original index that goes to position i
        // So to unreorder: output[order[i]] = input[i]
        size_t in_off  = (size_t)vec_idx * n + tid;
        size_t out_off = (size_t)vec_idx * n + order[tid];
        output[out_off] = input[in_off];
    }
}

// ============================================================================
// Banded Quantization Kernel
// ============================================================================
//
// Each thread block processes one vector.
// Band 0..n_bands-1 each get: fp16 scale + packed n-bit integers.
//
// Thread assignment: threads 0..band_size-1 handle band b.
// Phase 1: parallel reduction to find max(abs) per band → scale
// Phase 2: each thread quantizes its element and writes packed bits
//
// Note: bit packing is serialized within a band for correctness.
// A production kernel would use warp-level shuffle for parallel packing.

// Simplified kernel: one thread per vector, processes all bands sequentially.
// This is correct and fast enough for single-token decode.
// Batch kernels for prefill use the parallel version below.

__global__ void kernel_band_quantize_simple(
    const float *input,   // [n_vecs][n] VHT2 coefficients
    uint8_t *output,      // [n_vecs][total_bytes] packed output
    int n,                // head_dim
    int n_bands,
    const int *band_bits, // [n_bands] bits per band (device memory)
    int band_size,        // n / n_bands
    int total_bytes,      // bytes per compressed vector
    int n_vecs            // number of vectors
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n_vecs) return;

    const float *vec = input + (size_t)vec_idx * n;
    uint8_t *out     = output + (size_t)vec_idx * total_bytes;

    int offset = 0;

    for (int b = 0; b < n_bands; b++) {
        int band_off = b * band_size;
        int band_sz  = (b == n_bands - 1) ? (n - band_off) : band_size;
        const float *band = vec + band_off;
        int bits = band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;

        // Find max absolute value
        float amax = 0.0f;
        for (int i = 0; i < band_sz; i++) {
            float a = fabsf(band[i]);
            if (a > amax) amax = a;
        }

        float scale = (amax > 0.0f) ? amax / (float)max_val : 0.0f;

        // Store scale as fp16 first, then recompute inv_scale from the
        // stored fp16 value. This eliminates the quantize/dequantize
        // asymmetry: dequantize reads scale from fp16, so quantize must
        // use the same (rounded) value. Without this, variance-ranked
        // coefficients in band 0 (highest information) accumulate a
        // systematic error that shows up as PPL regression.
        __half scale_h = __float2half(scale);
        unsigned short scale_bits;
        memcpy(&scale_bits, &scale_h, sizeof(unsigned short));
        out[offset]     = scale_bits & 0xFF;
        out[offset + 1] = (scale_bits >> 8) & 0xFF;
        offset += 2;

        float scale_stored = __half2float(scale_h);
        float inv_scale = (scale_stored > 0.0f) ? 1.0f / scale_stored : 0.0f;

        // Pack quantized values
        unsigned long long bit_buffer = 0;
        int bit_pos = 0;

        for (int i = 0; i < band_sz; i++) {
            int q = __float2int_rn(band[i] * inv_scale);
            if (q > max_val)  q = max_val;
            if (q < -max_val) q = -max_val;

            unsigned int u = (unsigned int)(q + max_val);
            bit_buffer |= ((unsigned long long)u << bit_pos);
            bit_pos += bits;

            while (bit_pos >= 8) {
                out[offset++] = (uint8_t)(bit_buffer & 0xFF);
                bit_buffer >>= 8;
                bit_pos -= 8;
            }
        }
        if (bit_pos > 0) {
            out[offset++] = (uint8_t)(bit_buffer & 0xFF);
        }
    }
}

// ============================================================================
// Banded Dequantization Kernel
// ============================================================================

__global__ void kernel_band_dequantize_simple(
    const uint8_t *input, // [n_vecs][total_bytes]
    float *output,        // [n_vecs][n]
    int n,
    int n_bands,
    const int *band_bits,
    int band_size,
    int total_bytes,
    int n_vecs
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n_vecs) return;

    const uint8_t *in = input + (size_t)vec_idx * total_bytes;
    float *vec        = output + (size_t)vec_idx * n;

    int offset = 0;

    for (int b = 0; b < n_bands; b++) {
        int band_off = b * band_size;
        int band_sz  = (b == n_bands - 1) ? (n - band_off) : band_size;
        float *band = vec + band_off;
        int bits = band_bits[b];
        int max_val = (1 << (bits - 1)) - 1;
        unsigned int mask = (1u << bits) - 1;

        // Read scale. Mirror the CPU core's non-finite guard: if the fp16 scale
        // round-trips to +/-Inf or NaN (amax overflowed fp16 range on encode),
        // zero the band so the inverse VHT2 stays finite instead of cascading
        // NaN through the attention path.
        unsigned short scale_bits = (unsigned short)in[offset] |
                                    ((unsigned short)in[offset + 1] << 8);
        __half scale_h;
        memcpy(&scale_h, &scale_bits, sizeof(__half));
        float scale = __half2float(scale_h);
        if (!isfinite(scale)) scale = 0.0f;
        offset += 2;

        // Unpack
        unsigned long long bit_buffer = 0;
        int bit_pos = 0;
        int byte_idx = offset;

        for (int i = 0; i < band_sz; i++) {
            while (bit_pos < bits) {
                bit_buffer |= ((unsigned long long)in[byte_idx++] << bit_pos);
                bit_pos += 8;
            }

            unsigned int u = (unsigned int)(bit_buffer & mask);
            bit_buffer >>= bits;
            bit_pos -= bits;

            int q = (int)u - max_val;
            band[i] = (float)q * scale;
        }

        int data_bits = band_sz * bits;
        offset += (data_bits + 7) / 8;
    }
}

// ============================================================================
// Host-side launcher functions
// ============================================================================

extern "C" {

#include "shannon_prime_cuda.h"

// Forward declaration of the staged Hartley kernel launcher from
// shannon_prime_sqfree.cu. Used when n has a prime factor > 2.
extern "C" void sp_cuda_vilenkin_inplace(float *d_data, int pad_dim, int n_vecs,
                                          int *d_factors, int n_factors,
                                          cudaStream_t stream);

// Factor n into {2,3,5,7,11}. Returns the number of factors or -1 if n has
// a prime factor > 11 (in which case the caller should sqfree_pad first).
static int _sp_cuda_factor_small(int n, int *factors_out, int max_factors) {
    static const int primes[] = {2, 3, 5, 7, 11};
    int nf = 0;
    int d = n;
    for (int i = 0; i < 5; i++) {
        while (d % primes[i] == 0 && nf < max_factors) {
            factors_out[nf++] = primes[i];
            d /= primes[i];
        }
    }
    return (d == 1) ? nf : -1;
}

// Public VHT2 forward — self-inverse, dispatches p=2 butterfly when n is a
// power of 2, otherwise the staged Vilenkin kernel for sqfree-padded dims.
void sp_cuda_vht2_forward(float *d_data, int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    if (n > 0 && (n & (n - 1)) == 0) {
        // Power of 2: fast butterfly with per-stage 1/√2 normalisation
        int smem_bytes = n * sizeof(float);
        kernel_vht2_p2<<<n_vecs, n, smem_bytes, s>>>(d_data, n);
        return;
    }
    // General path via the sqfree staged Hartley kernel
    int factors[16];
    int nf = _sp_cuda_factor_small(n, factors, 16);
    if (nf <= 0) {
        // Unsupported dim — leave data unchanged so the problem is loud
        fprintf(stderr, "[sp_cuda_vht2_forward] dim %d doesn't factor into "
                        "{2,3,5,7,11}; use sqfree_pad_dim first\n", n);
        return;
    }
    int *d_factors = NULL;
    cudaMalloc(&d_factors, nf * sizeof(int));
    cudaMemcpyAsync(d_factors, factors, nf * sizeof(int),
                    cudaMemcpyHostToDevice, s);
    sp_cuda_vilenkin_inplace(d_data, n, n_vecs, d_factors, nf, s);
    cudaStreamSynchronize(s);
    cudaFree(d_factors);
}

void sp_cuda_mobius_reorder(float *d_data, const int *d_order,
                            int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    // Need a temporary buffer (in-place reorder isn't safe)
    float *d_tmp;
    cudaMalloc(&d_tmp, (size_t)n_vecs * n * sizeof(float));
    kernel_mobius_reorder<<<n_vecs, n, 0, s>>>(d_data, d_tmp, d_order, n);
    cudaMemcpyAsync(d_data, d_tmp, (size_t)n_vecs * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    cudaFree(d_tmp);
}

void sp_cuda_mobius_unreorder(float *d_data, const int *d_order,
                              int n, int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    float *d_tmp;
    cudaMalloc(&d_tmp, (size_t)n_vecs * n * sizeof(float));
    kernel_mobius_unreorder<<<n_vecs, n, 0, s>>>(d_data, d_tmp, d_order, n);
    cudaMemcpyAsync(d_data, d_tmp, (size_t)n_vecs * n * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    cudaFree(d_tmp);
}

void sp_cuda_band_quantize(const float *d_input, void *d_output,
                           const sp_band_config_t *bc,
                           int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    // Must be the logical head_dim, not band_size * n_bands. The kernel
    // uses `n` as both the per-vector stride AND the last-band upper
    // bound, so when head_dim % n_bands != 0 (e.g. 10 bands @ hd=128)
    // multiplying silently orphans the tail coefficients.
    int n = bc->head_dim;

    // Upload band_bits to device
    int *d_band_bits;
    cudaMalloc(&d_band_bits, bc->n_bands * sizeof(int));
    cudaMemcpyAsync(d_band_bits, bc->band_bits, bc->n_bands * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    // One thread per vector (simple version for decode)
    // For large n_vecs (prefill), launch more threads
    int block = 256;
    int grid  = (n_vecs + block - 1) / block;
    kernel_band_quantize_simple<<<grid, block, 0, s>>>(
        d_input, (uint8_t *)d_output,
        n, bc->n_bands, d_band_bits,
        bc->band_size, bc->total_bytes, n_vecs
    );

    cudaFree(d_band_bits);
}

void sp_cuda_band_dequantize(const void *d_input, float *d_output,
                             const sp_band_config_t *bc,
                             int n_vecs, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int n = bc->head_dim;  // see comment in sp_cuda_band_quantize

    int *d_band_bits;
    cudaMalloc(&d_band_bits, bc->n_bands * sizeof(int));
    cudaMemcpyAsync(d_band_bits, bc->band_bits, bc->n_bands * sizeof(int),
                    cudaMemcpyHostToDevice, s);

    int block = 256;
    int grid  = (n_vecs + block - 1) / block;
    kernel_band_dequantize_simple<<<grid, block, 0, s>>>(
        (const uint8_t *)d_input, d_output,
        n, bc->n_bands, d_band_bits,
        bc->band_size, bc->total_bytes, n_vecs
    );

    cudaFree(d_band_bits);
}

// ============================================================================
// Shadow cache implementation
// ============================================================================

int sp_cuda_cache_init(sp_cuda_cache_t *cc, const sp_config_t *cfg,
                       int max_seq_len, void *stream) {
    memcpy(&cc->config, cfg, sizeof(sp_config_t));
    cc->max_seq_len = max_seq_len;
    cc->stream = stream;

    sp_band_config_init(&cc->k_bands, cfg->head_dim,
                        cfg->k_n_bands, cfg->k_band_bits);
    sp_band_config_init(&cc->v_bands, cfg->head_dim,
                        cfg->v_n_bands, cfg->v_band_bits);

    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    size_t k_total = (size_t)n_slots * max_seq_len * cc->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * max_seq_len * cc->v_bands.total_bytes;

    cudaMalloc(&cc->d_k_cache, k_total);
    cudaMalloc(&cc->d_v_cache, v_total);
    cudaMalloc(&cc->d_scratch, cfg->head_dim * sizeof(float));

    // Upload Möbius tables
    if (cfg->use_mobius_mask) {
        sp_mobius_mask_t mask;
        sp_mobius_mask_init(&mask, cfg->head_dim);

        cudaMalloc(&cc->d_mobius_order, cfg->head_dim * sizeof(int));
        cudaMemcpy(cc->d_mobius_order, mask.order,
                   cfg->head_dim * sizeof(int), cudaMemcpyHostToDevice);

        // Build inverse permutation for unreorder
        int *inv = (int *)malloc(cfg->head_dim * sizeof(int));
        for (int i = 0; i < cfg->head_dim; i++) {
            inv[mask.order[i]] = i;
        }
        cudaMalloc(&cc->d_mobius_inv, cfg->head_dim * sizeof(int));
        cudaMemcpy(cc->d_mobius_inv, inv,
                   cfg->head_dim * sizeof(int), cudaMemcpyHostToDevice);

        free(inv);
        sp_mobius_mask_free(&mask);
    } else {
        cc->d_mobius_order = NULL;
        cc->d_mobius_inv   = NULL;
    }

    fprintf(stderr, "[Shannon-Prime CUDA] Cache allocated:\n");
    fprintf(stderr, "  K: %.2f MB (%d bytes/vec × %d slots × %d seq)\n",
            k_total / (1024.0 * 1024.0), cc->k_bands.total_bytes,
            n_slots, max_seq_len);
    fprintf(stderr, "  V: %.2f MB (%d bytes/vec × %d slots × %d seq)\n",
            v_total / (1024.0 * 1024.0), cc->v_bands.total_bytes,
            n_slots, max_seq_len);
    fprintf(stderr, "  Total: %.2f MB (vs fp16 baseline: %.2f MB)\n",
            (k_total + v_total) / (1024.0 * 1024.0),
            (size_t)n_slots * max_seq_len * cfg->head_dim * 2 * 2 / (1024.0 * 1024.0));

    return 0;
}

void sp_cuda_cache_free(sp_cuda_cache_t *cc) {
    cudaFree(cc->d_k_cache);
    cudaFree(cc->d_v_cache);
    cudaFree(cc->d_scratch);
    cudaFree(cc->d_mobius_order);
    cudaFree(cc->d_mobius_inv);
}

// Single-vector write: VHT2 → Möbius → quantize → store
void sp_cuda_write_k(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_k_vec) {
    int hd = cc->config.head_dim;
    float *scratch = cc->d_scratch;
    cudaStream_t s = (cudaStream_t)cc->stream;

    // Copy to scratch
    cudaMemcpyAsync(scratch, d_k_vec, hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    // VHT2 forward
    sp_cuda_vht2_forward(scratch, hd, 1, cc->stream);

    // Möbius reorder
    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_reorder(scratch, cc->d_mobius_order, hd, 1, cc->stream);
    }

    // Quantize into cache
    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                  + (size_t)pos * cc->k_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_k_cache + offset;

    sp_cuda_band_quantize(scratch, dest, &cc->k_bands, 1, cc->stream);
}

void sp_cuda_write_v(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_v_vec) {
    int hd = cc->config.head_dim;
    float *scratch = cc->d_scratch;
    cudaStream_t s = (cudaStream_t)cc->stream;

    cudaMemcpyAsync(scratch, d_v_vec, hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    sp_cuda_vht2_forward(scratch, hd, 1, cc->stream);

    // No Möbius for V (uniform spectrum)

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                  + (size_t)pos * cc->v_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_v_cache + offset;

    sp_cuda_band_quantize(scratch, dest, &cc->v_bands, 1, cc->stream);
}

// Single-vector read: load → dequantize → Möbius unreorder → VHT2 (self-inverse)
void sp_cuda_read_k(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_k_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                  + (size_t)pos * cc->k_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_k_cache + offset;

    sp_cuda_band_dequantize(src, d_k_out, &cc->k_bands, 1, (void *)s);

    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_unreorder(d_k_out, cc->d_mobius_order, hd, 1, (void *)s);
    }

    sp_cuda_vht2_forward(d_k_out, hd, 1, (void *)s);
}

void sp_cuda_read_v(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_v_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                  + (size_t)pos * cc->v_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_v_cache + offset;

    sp_cuda_band_dequantize(src, d_v_out, &cc->v_bands, 1, (void *)s);

    // No Möbius unreorder for V
    sp_cuda_vht2_forward(d_v_out, hd, 1, (void *)s);
}

// ============================================================================
// Batch operations for prefill
// ============================================================================

void sp_cuda_write_k_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_k_vecs) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    // Allocate batch scratch
    float *d_work;
    cudaMalloc(&d_work, (size_t)n_pos * hd * sizeof(float));
    cudaMemcpyAsync(d_work, d_k_vecs, (size_t)n_pos * hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    // VHT2 all vectors in parallel
    sp_cuda_vht2_forward(d_work, hd, n_pos, cc->stream);

    // Möbius reorder all
    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_reorder(d_work, cc->d_mobius_order, hd, n_pos, cc->stream);
    }

    // Quantize all into cache
    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                       + (size_t)start_pos * cc->k_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_k_cache + base_offset;

    sp_cuda_band_quantize(d_work, dest, &cc->k_bands, n_pos, cc->stream);

    cudaFree(d_work);
}

void sp_cuda_write_v_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_v_vecs) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    float *d_work;
    cudaMalloc(&d_work, (size_t)n_pos * hd * sizeof(float));
    cudaMemcpyAsync(d_work, d_v_vecs, (size_t)n_pos * hd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);

    sp_cuda_vht2_forward(d_work, hd, n_pos, cc->stream);
    // No Möbius for V

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                       + (size_t)start_pos * cc->v_bands.total_bytes;
    void *dest = (uint8_t *)cc->d_v_cache + base_offset;

    sp_cuda_band_quantize(d_work, dest, &cc->v_bands, n_pos, cc->stream);

    cudaFree(d_work);
}

void sp_cuda_read_k_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_k_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->k_bands.total_bytes
                       + (size_t)start_pos * cc->k_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_k_cache + base_offset;

    sp_cuda_band_dequantize(src, d_k_out, &cc->k_bands, n_pos, (void *)s);

    if (cc->config.use_mobius_mask) {
        sp_cuda_mobius_unreorder(d_k_out, cc->d_mobius_order, hd, n_pos, (void *)s);
    }

    sp_cuda_vht2_forward(d_k_out, hd, n_pos, (void *)s);
}

void sp_cuda_read_v_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_v_out) {
    int hd = cc->config.head_dim;
    cudaStream_t s = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    size_t base_offset = (size_t)slot * cc->max_seq_len * cc->v_bands.total_bytes
                       + (size_t)start_pos * cc->v_bands.total_bytes;
    const void *src = (const uint8_t *)cc->d_v_cache + base_offset;

    sp_cuda_band_dequantize(src, d_v_out, &cc->v_bands, n_pos, (void *)s);
    sp_cuda_vht2_forward(d_v_out, hd, n_pos, (void *)s);
}

// ============================================================================
// Diagnostics
// ============================================================================

void sp_cuda_print_memory(const sp_cuda_cache_t *cc) {
    int n_slots = cc->config.n_layers * cc->config.n_heads_kv;
    size_t k_total = (size_t)n_slots * cc->max_seq_len * cc->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * cc->max_seq_len * cc->v_bands.total_bytes;
    size_t baseline = (size_t)n_slots * cc->max_seq_len * cc->config.head_dim * 2 * 2;

    fprintf(stderr, "[Shannon-Prime CUDA] Memory:\n");
    fprintf(stderr, "  Compressed: %.2f MB (K: %.2f + V: %.2f)\n",
            (k_total + v_total) / (1024.0 * 1024.0),
            k_total / (1024.0 * 1024.0),
            v_total / (1024.0 * 1024.0));
    fprintf(stderr, "  Baseline:   %.2f MB\n", baseline / (1024.0 * 1024.0));
    fprintf(stderr, "  Ratio:      %.1f×\n",
            (double)baseline / (double)(k_total + v_total));
}

// ============================================================================
// Cold storage — tiered GPU ↔ CPU offload
// ============================================================================

void sp_cuda_d2h_async(void *dst_host, const void *src_device,
                       int64_t n_bytes, void *stream) {
    cudaMemcpyAsync(dst_host, src_device, (size_t)n_bytes,
                    cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

void sp_cuda_h2d_async(void *dst_device, const void *src_host,
                       int64_t n_bytes, void *stream) {
    cudaMemcpyAsync(dst_device, src_host, (size_t)n_bytes,
                    cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

int sp_cuda_cold_layer_init(sp_cuda_cold_layer_t *cl,
                            int64_t cap_bytes,
                            int packed_stride, int n_heads) {
    if (!cl || cap_bytes <= 0 || packed_stride <= 0 || n_heads <= 0)
        return -1;
    memset(cl, 0, sizeof(*cl));

    cudaError_t err = cudaHostAlloc(&cl->h_pinned, (size_t)cap_bytes,
                                    cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "[sp-cold] cudaHostAlloc failed (%lld bytes): %s\n",
                (long long)cap_bytes, cudaGetErrorString(err));
        return -1;
    }
    cl->capacity      = cap_bytes;
    cl->packed_stride  = packed_stride;
    cl->n_heads        = n_heads;
    cl->write_head     = 0;
    cl->oldest_pos     = -1;
    cl->newest_pos     = -1;
    cl->ring_mode      = false;
    cl->initialized    = true;
    return 0;
}

void sp_cuda_cold_layer_free(sp_cuda_cold_layer_t *cl) {
    if (!cl) return;
    if (cl->h_pinned) {
        cudaFreeHost(cl->h_pinned);
        cl->h_pinned = NULL;
    }
    cl->initialized = false;
}

int sp_cuda_cold_writeback(sp_cuda_cold_layer_t *cl,
                           const void *d_packed,
                           int start_pos, int n_pos,
                           void *stream) {
    if (!cl || !cl->initialized || !d_packed || n_pos <= 0) return -1;

    const int64_t bytes_per_pos = (int64_t)cl->n_heads * cl->packed_stride;
    const int64_t new_bytes     = (int64_t)n_pos * bytes_per_pos;
    const int64_t src_offset    = (int64_t)start_pos * bytes_per_pos;

    // Ring-buffer wrap check
    if (cl->write_head + new_bytes > cl->capacity) {
        cl->write_head = 0;
        cl->ring_mode  = true;
    }

    sp_cuda_d2h_async(cl->h_pinned + cl->write_head,
                      (const uint8_t*)d_packed + src_offset,
                      new_bytes, stream);

    cl->write_head += new_bytes;
    cl->newest_pos  = start_pos + n_pos - 1;

    if (cl->ring_mode) {
        const int64_t max_positions = cl->capacity / bytes_per_pos;
        cl->oldest_pos = cl->newest_pos - max_positions + 1;
        if (cl->oldest_pos < 0) cl->oldest_pos = 0;
    } else {
        if (cl->oldest_pos < 0) cl->oldest_pos = 0;
    }

    return 0;
}

int sp_cuda_cold_restore(const sp_cuda_cold_layer_t *cl,
                         void *d_packed,
                         int n_pos,
                         void *stream) {
    if (!cl || !cl->initialized || !d_packed || n_pos <= 0) return -1;
    if (cl->ring_mode) {
        fprintf(stderr, "[sp-cold] restore from ring-mode buffer not supported "
                "(non-linear layout)\n");
        return -1;
    }

    const int64_t bytes_per_pos = (int64_t)cl->n_heads * cl->packed_stride;
    const int64_t total_bytes   = (int64_t)n_pos * bytes_per_pos;

    if (total_bytes > cl->capacity) {
        fprintf(stderr, "[sp-cold] restore: requested %lld bytes > capacity %lld\n",
                (long long)total_bytes, (long long)cl->capacity);
        return -1;
    }

    sp_cuda_h2d_async(d_packed, cl->h_pinned, total_bytes, stream);
    return 0;
}

} // extern "C"
