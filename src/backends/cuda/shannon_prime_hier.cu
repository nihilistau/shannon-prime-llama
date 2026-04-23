// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CUDA kernels for the hierarchical Vilenkin predictor GPU-resident cache.
//
// The hierarchical path achieves maximum compression by storing a small
// Kronecker skeleton (~9% of pad_dim) and predicting the remaining
// coefficients via a per-(layer,head) linear map W. This file provides
// GPU-resident compress/decompress that keeps everything in VRAM.
//
// Pipeline (write):
//   raw vec → sqfree pad → Vilenkin forward → gather skeleton
//   → band-quantize skeleton → W·skeleton (prediction)
//   → deviation = actual_target - predicted → magnitude
//   → quantize residual → pack bits → store
//
// Pipeline (read):
//   load → band-dequantize skeleton → W·skeleton (prediction)
//   → read magnitude → unpack residual bits → dequantize residual
//   → scatter skeleton + (predicted + residual) → inverse Vilenkin
//   → sqfree unpad → output
//
// Reuses existing CUDA building blocks from shannon_prime_sqfree.cu:
//   kernel_sqfree_pad / kernel_sqfree_unpad
//   sp_cuda_vilenkin_inplace
//   kernel_gather / kernel_scatter
//   sp_cuda_band_quantize / sp_cuda_band_dequantize

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../core/shannon_prime.h"

// External declarations for shared kernels (defined in shannon_prime_sqfree.cu)
extern "C" void sp_cuda_vilenkin_inplace(float *d_data, int pad_dim, int n_vecs,
                                          int *d_factors, int n_factors,
                                          cudaStream_t stream);
extern __global__ void kernel_sqfree_pad(const float *in, float *out,
                                          int hd, int pd, int n_vecs);
extern __global__ void kernel_sqfree_unpad(const float *in, float *out,
                                            int head_dim, int pad_dim, int n_vecs);
extern __global__ void kernel_gather(const float *in, const int *idx,
                                      float *out, int n);
extern __global__ void kernel_scatter(const float *in, const int *idx,
                                       float *out, int n);

// Forward declarations for band quantize/dequantize launchers
extern "C" {
void sp_cuda_band_quantize(const float *d_input, void *d_output,
                           const sp_band_config_t *bc,
                           int n_vecs, void *stream);
void sp_cuda_band_dequantize(const void *d_input, float *d_output,
                             const sp_band_config_t *bc,
                             int n_vecs, void *stream);
}

// ============================================================================
// Hierarchical prediction kernel — target = W · skeleton (fp16 W)
// ============================================================================
//
// One thread per target coefficient. W is stored row-major:
//   W_slot[t * n_skeleton + s] (fp16)
// target_out[t] = Σ_s W[t * n_skeleton + s] · skeleton[s]

__global__ void kernel_hier_predict(
    const uint16_t *W,       // [n_target × n_skeleton] fp16 for this slot
    const float *skeleton,   // [n_skeleton] fp32
    float *target_out,       // [n_target] fp32
    int n_target,
    int n_skeleton
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_target) return;

    const uint16_t *Wrow = W + (size_t)t * n_skeleton;
    float acc = 0.0f;
    for (int s = 0; s < n_skeleton; s++) {
        __half wh;
        memcpy(&wh, &Wrow[s], sizeof(__half));
        acc += __half2float(wh) * skeleton[s];
    }
    target_out[t] = acc;
}

// ============================================================================
// Deviation kernel: dev[t] = coeffs[target_idx[t]] - predicted[t]
// ============================================================================

__global__ void kernel_hier_deviation(
    const float *coeffs,       // [pad_dim] all Vilenkin coefficients
    const int   *target_idx,   // [n_target] indices into coeffs
    const float *predicted,    // [n_target] predicted targets
    float *deviation,          // [n_target] output
    int n_target
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_target) return;
    deviation[t] = coeffs[target_idx[t]] - predicted[t];
}

// ============================================================================
// Residual magnitude: mean(|deviation|), with floor at 1e-12
// ============================================================================

// Single-block reduction. n_target is typically ~130, so one block suffices.
__global__ void kernel_residual_magnitude(
    const float *deviation,  // [n_target]
    float *mag_out,          // [1] output scalar
    int n_target
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // Accumulate |dev| across threads
    float local = 0.0f;
    for (int i = tid; i < n_target; i += blockDim.x) {
        local += fabsf(deviation[i]);
    }
    smem[tid] = local;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = smem[0] / (float)n_target;
        if (m < 1e-12f) m = 1e-12f;
        mag_out[0] = m;
    }
}

// ============================================================================
// Residual quantize: levels[i] = clamp(round(dev[i]/step + center), 0, L-1)
// where step = 2*mag/(L-1), center = (L-1)/2, L = 2^nbits
// ============================================================================

__global__ void kernel_quantize_residual(
    const float *deviation,     // [n_target]
    const float *mag,           // [1] magnitude scalar (device)
    unsigned char *levels,      // [n_target] output quantized levels
    int n_target,
    int nbits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_target) return;

    int L = 1 << nbits;
    float m = mag[0];
    float step = 2.0f * m / (float)(L - 1);
    float center = (float)(L - 1) / 2.0f;
    float inv_step = (step > 0.0f) ? 1.0f / step : 0.0f;

    int q = __float2int_rn(deviation[i] * inv_step + center);
    if (q < 0) q = 0;
    if (q >= L) q = L - 1;
    levels[i] = (unsigned char)q;
}

// ============================================================================
// Residual dequantize: val[i] = (levels[i] - center) * step
// ============================================================================

__global__ void kernel_dequantize_residual(
    const unsigned char *levels, // [n_target]
    const float *mag,            // [1] magnitude scalar (device)
    float *output,               // [n_target]
    int n_target,
    int nbits
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_target) return;

    int L = 1 << nbits;
    float m = mag[0];
    float step = 2.0f * m / (float)(L - 1);
    float center = (float)(L - 1) / 2.0f;

    output[i] = ((float)levels[i] - center) * step;
}

// ============================================================================
// Pack residual levels into bit-packed storage (in the slot's output buffer)
// ============================================================================

// Single-thread packing — n_target is small (~130), overhead is negligible
// vs the kernel launch cost. Runs on GPU to avoid a D2H round-trip.
__global__ void kernel_pack_residual_bits(
    const unsigned char *levels, // [n_target]
    unsigned char *out,          // output packed bytes
    int n_target,
    int nbits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int res_bytes = (n_target * nbits + 7) / 8;
    // Zero first
    for (int b = 0; b < res_bytes; b++) out[b] = 0;

    for (int i = 0; i < n_target; i++) {
        int bit_off = i * nbits;
        out[bit_off / 8] |= (levels[i] << (bit_off % 8));
        if ((bit_off % 8) + nbits > 8 && (bit_off / 8 + 1) < res_bytes) {
            out[bit_off / 8 + 1] |= (levels[i] >> (8 - bit_off % 8));
        }
    }
}

// ============================================================================
// Unpack residual bits into level array
// ============================================================================

__global__ void kernel_unpack_residual_bits(
    const unsigned char *in,     // packed bytes
    unsigned char *levels,       // [n_target] output
    int n_target,
    int nbits
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int L = 1 << nbits;
    int res_bytes = (n_target * nbits + 7) / 8;
    for (int i = 0; i < n_target; i++) {
        int bit_off = i * nbits;
        int val = (in[bit_off / 8] >> (bit_off % 8));
        if ((bit_off % 8) + nbits > 8 && (bit_off / 8 + 1) < res_bytes) {
            val |= (in[bit_off / 8 + 1] << (8 - bit_off % 8));
        }
        levels[i] = (unsigned char)(val & (L - 1));
    }
}

// ============================================================================
// Scatter predicted + residual to target positions
// ============================================================================

__global__ void kernel_hier_scatter_sum(
    const float *predicted,    // [n_target]
    const float *residual,     // [n_target]
    const int   *target_idx,   // [n_target]
    float *coeffs,             // [pad_dim] output (skeleton already scattered)
    int n_target
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_target) return;
    coeffs[target_idx[t]] = predicted[t] + residual[t];
}

// ============================================================================
// Host-side API: init / free / upload_W / write / read
// ============================================================================

extern "C" {
#include "shannon_prime_cuda.h"
}

extern "C" {

int sp_cuda_hier_cache_init(sp_cuda_hier_cache_t *hc,
                             const sp_config_t *cfg,
                             int pad_dim, int n_skeleton, int n_target,
                             int target_res_bits,
                             const int *skeleton_idx,
                             const int *target_idx,
                             const sp_band_config_t *skel_bands,
                             int max_seq_len, int n_slots,
                             void *stream) {
    memset(hc, 0, sizeof(*hc));
    hc->config          = *cfg;
    hc->pad_dim         = pad_dim;
    hc->n_skeleton      = n_skeleton;
    hc->n_target        = n_target;
    hc->target_res_bits = target_res_bits;
    hc->n_slots         = n_slots;
    hc->max_seq_len     = max_seq_len;
    hc->skel_bands      = *skel_bands;
    hc->stream          = stream;

    cudaStream_t s = (cudaStream_t)stream;

    // Upload index arrays
    cudaMalloc(&hc->d_skeleton_idx, (size_t)n_skeleton * sizeof(int));
    cudaMemcpy(hc->d_skeleton_idx, skeleton_idx,
               (size_t)n_skeleton * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&hc->d_target_idx, (size_t)n_target * sizeof(int));
    cudaMemcpy(hc->d_target_idx, target_idx,
               (size_t)n_target * sizeof(int), cudaMemcpyHostToDevice);

    // Upload skel band bits
    cudaMalloc(&hc->d_skel_band_bits, (size_t)skel_bands->n_bands * sizeof(int));
    cudaMemcpy(hc->d_skel_band_bits, skel_bands->band_bits,
               (size_t)skel_bands->n_bands * sizeof(int), cudaMemcpyHostToDevice);

    // Vilenkin factor list
    int factors[16];
    int nf = 0;
    {
        int residue = pad_dim;
        const int primes[] = {2, 3, 5, 7, 11};
        for (unsigned p = 0; p < sizeof(primes) / sizeof(primes[0]); ++p) {
            while (residue % primes[p] == 0) {
                factors[nf++] = primes[p];
                residue /= primes[p];
            }
        }
        if (residue != 1) {
            fprintf(stderr, "[sp-cuda-hier] pad_dim %d doesn't factor over "
                            "{2,3,5,7,11}; init failed\n", pad_dim);
            return -1;
        }
    }
    hc->n_factors = nf;
    cudaMalloc(&hc->d_vilenkin_factors, (size_t)nf * sizeof(int));
    cudaMemcpy(hc->d_vilenkin_factors, factors,
               (size_t)nf * sizeof(int), cudaMemcpyHostToDevice);

    // Compressed storage layout per position:
    //   [skel_bands.total_bytes] + [4 bytes magnitude] + [res_bytes]
    int res_bytes = (n_target * target_res_bits + 7) / 8;
    hc->bytes_per_pos_k = skel_bands->total_bytes + 4 + res_bytes;
    hc->bytes_per_pos_v = skel_bands->total_bytes + 4 + res_bytes;

    size_t cache_bytes_k = (size_t)n_slots * max_seq_len * hc->bytes_per_pos_k;
    size_t cache_bytes_v = (size_t)n_slots * max_seq_len * hc->bytes_per_pos_v;
    cudaMalloc(&hc->d_k_cache, cache_bytes_k);
    cudaMalloc(&hc->d_v_cache, cache_bytes_v);
    cudaMemsetAsync(hc->d_k_cache, 0, cache_bytes_k, s);
    cudaMemsetAsync(hc->d_v_cache, 0, cache_bytes_v, s);

    // W matrices — allocate but don't fill until upload_W
    size_t w_bytes = (size_t)n_slots * n_target * n_skeleton * sizeof(uint16_t);
    cudaMalloc(&hc->d_W, w_bytes);
    cudaMemsetAsync(hc->d_W, 0, w_bytes, s);

    // Scratch buffers
    cudaMalloc(&hc->d_pad_scratch,    (size_t)pad_dim * sizeof(float));
    cudaMalloc(&hc->d_coeff_scratch,  (size_t)pad_dim * sizeof(float));
    cudaMalloc(&hc->d_skel_scratch,   (size_t)n_skeleton * sizeof(float));
    cudaMalloc(&hc->d_pred_scratch,   (size_t)n_target * sizeof(float));
    cudaMalloc(&hc->d_dev_scratch,    (size_t)n_target * sizeof(float));
    cudaMalloc(&hc->d_mag_scratch,    sizeof(float));
    cudaMalloc(&hc->d_levels_scratch, (size_t)n_target);

    fprintf(stderr, "[sp-cuda-hier] init: pd=%d skel=%d target=%d res_bits=%d "
                    "bytes_per_pos=%d n_slots=%d max_seq=%d\n",
            pad_dim, n_skeleton, n_target, target_res_bits,
            hc->bytes_per_pos_k, n_slots, max_seq_len);
    return 0;
}

void sp_cuda_hier_cache_free(sp_cuda_hier_cache_t *hc) {
    if (hc->d_W)                cudaFree(hc->d_W);
    if (hc->d_skeleton_idx)     cudaFree(hc->d_skeleton_idx);
    if (hc->d_target_idx)       cudaFree(hc->d_target_idx);
    if (hc->d_skel_band_bits)   cudaFree(hc->d_skel_band_bits);
    if (hc->d_vilenkin_factors) cudaFree(hc->d_vilenkin_factors);
    if (hc->d_k_cache)          cudaFree(hc->d_k_cache);
    if (hc->d_v_cache)          cudaFree(hc->d_v_cache);
    if (hc->d_pad_scratch)      cudaFree(hc->d_pad_scratch);
    if (hc->d_coeff_scratch)    cudaFree(hc->d_coeff_scratch);
    if (hc->d_skel_scratch)     cudaFree(hc->d_skel_scratch);
    if (hc->d_pred_scratch)     cudaFree(hc->d_pred_scratch);
    if (hc->d_dev_scratch)      cudaFree(hc->d_dev_scratch);
    if (hc->d_mag_scratch)      cudaFree(hc->d_mag_scratch);
    if (hc->d_levels_scratch)   cudaFree(hc->d_levels_scratch);
    memset(hc, 0, sizeof(*hc));
}

int sp_cuda_hier_cache_upload_W(sp_cuda_hier_cache_t *hc,
                                 const uint16_t *W_all) {
    size_t w_bytes = (size_t)hc->n_slots * hc->n_target * hc->n_skeleton
                   * sizeof(uint16_t);
    cudaError_t err = cudaMemcpy(hc->d_W, W_all, w_bytes,
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[sp-cuda-hier] upload_W failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// ── Internal write helper ─────────────────────────────────────────────

static void sp_cuda_hier_write_one(sp_cuda_hier_cache_t *hc,
                                    int layer, int head, int pos,
                                    const float *d_vec,
                                    unsigned char *d_cache,
                                    int bytes_per_pos)
{
    const int hd      = hc->config.head_dim;
    const int pd      = hc->pad_dim;
    const int ns      = hc->n_skeleton;
    const int nt      = hc->n_target;
    const int nbits   = hc->target_res_bits;
    cudaStream_t s    = (cudaStream_t)hc->stream;

    int slot = layer * hc->config.n_heads_kv + head;
    unsigned char *dest = d_cache
        + (size_t)slot * hc->max_seq_len * bytes_per_pos
        + (size_t)pos  * bytes_per_pos;

    // 1. Sqfree pad → d_pad_scratch
    kernel_sqfree_pad<<<1, 256, 0, s>>>(d_vec, hc->d_pad_scratch, hd, pd, 1);

    // 2. Vilenkin forward → d_coeff_scratch
    cudaMemcpyAsync(hc->d_coeff_scratch, hc->d_pad_scratch,
                    (size_t)pd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    sp_cuda_vilenkin_inplace(hc->d_coeff_scratch, pd, 1,
                              hc->d_vilenkin_factors, hc->n_factors, s);

    // 3. Gather skeleton → d_skel_scratch
    {
        int blk = 128;
        int grid = (ns + blk - 1) / blk;
        kernel_gather<<<grid, blk, 0, s>>>(
            hc->d_coeff_scratch, hc->d_skeleton_idx,
            hc->d_skel_scratch, ns);
    }

    // 4. Band-quantize skeleton into output slot
    sp_cuda_band_quantize(hc->d_skel_scratch, dest,
                          &hc->skel_bands, 1, s);

    // 5. Predict targets: pred = W_slot · skeleton
    {
        const uint16_t *W_slot = hc->d_W
            + (size_t)slot * nt * ns;
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_hier_predict<<<grid, blk, 0, s>>>(
            W_slot, hc->d_skel_scratch, hc->d_pred_scratch,
            nt, ns);
    }

    // 6. Deviation: dev[t] = coeff[target_idx[t]] - pred[t]
    {
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_hier_deviation<<<grid, blk, 0, s>>>(
            hc->d_coeff_scratch, hc->d_target_idx,
            hc->d_pred_scratch, hc->d_dev_scratch, nt);
    }

    // 7. Residual magnitude
    {
        int blk = 256;
        int smem = blk * sizeof(float);
        kernel_residual_magnitude<<<1, blk, smem, s>>>(
            hc->d_dev_scratch, hc->d_mag_scratch, nt);
    }

    // 8. Store magnitude (4 bytes) after skeleton bytes
    unsigned char *mag_dest = dest + hc->skel_bands.total_bytes;
    cudaMemcpyAsync(mag_dest, hc->d_mag_scratch, 4,
                    cudaMemcpyDeviceToDevice, s);

    // 9. Quantize residuals
    {
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_quantize_residual<<<grid, blk, 0, s>>>(
            hc->d_dev_scratch, hc->d_mag_scratch,
            hc->d_levels_scratch, nt, nbits);
    }

    // 10. Pack residual bits into output
    unsigned char *res_dest = mag_dest + 4;
    kernel_pack_residual_bits<<<1, 1, 0, s>>>(
        hc->d_levels_scratch, res_dest, nt, nbits);
}

// ── Internal read helper ──────────────────────────────────────────────

static void sp_cuda_hier_read_one(const sp_cuda_hier_cache_t *hc,
                                   int layer, int head, int pos,
                                   float *d_vec_out,
                                   const unsigned char *d_cache,
                                   int bytes_per_pos)
{
    const int hd      = hc->config.head_dim;
    const int pd      = hc->pad_dim;
    const int ns      = hc->n_skeleton;
    const int nt      = hc->n_target;
    const int nbits   = hc->target_res_bits;
    cudaStream_t s    = (cudaStream_t)hc->stream;

    // Cast away const on scratch pointers — callers serialise access
    sp_cuda_hier_cache_t *hc_mut = (sp_cuda_hier_cache_t *)hc;

    int slot = layer * hc->config.n_heads_kv + head;
    const unsigned char *src = d_cache
        + (size_t)slot * hc->max_seq_len * bytes_per_pos
        + (size_t)pos  * bytes_per_pos;

    // 1. Band-dequantize skeleton → d_skel_scratch
    sp_cuda_band_dequantize(src, hc_mut->d_skel_scratch,
                            &hc->skel_bands, 1, (void *)s);

    // 2. Predict targets from skeleton
    {
        const uint16_t *W_slot = hc->d_W
            + (size_t)slot * nt * ns;
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_hier_predict<<<grid, blk, 0, s>>>(
            W_slot, hc_mut->d_skel_scratch, hc_mut->d_pred_scratch,
            nt, ns);
    }

    // 3. Read magnitude
    const unsigned char *mag_src = src + hc->skel_bands.total_bytes;
    cudaMemcpyAsync(hc_mut->d_mag_scratch, mag_src, 4,
                    cudaMemcpyDeviceToDevice, s);

    // 4. Unpack residual bits → levels
    const unsigned char *res_src = mag_src + 4;
    kernel_unpack_residual_bits<<<1, 1, 0, s>>>(
        res_src, hc_mut->d_levels_scratch, nt, nbits);

    // 5. Dequantize residual → d_dev_scratch
    {
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_dequantize_residual<<<grid, blk, 0, s>>>(
            hc_mut->d_levels_scratch, hc_mut->d_mag_scratch,
            hc_mut->d_dev_scratch, nt, nbits);
    }

    // 6. Reconstruct full coefficient vector
    //    Zero coeff_scratch, scatter skeleton, scatter pred+residual
    cudaMemsetAsync(hc_mut->d_coeff_scratch, 0,
                    (size_t)pd * sizeof(float), s);
    {
        int blk = 128;
        int grid = (ns + blk - 1) / blk;
        kernel_scatter<<<grid, blk, 0, s>>>(
            hc_mut->d_skel_scratch, hc_mut->d_skeleton_idx,
            hc_mut->d_coeff_scratch, ns);
    }
    {
        int blk = 128;
        int grid = (nt + blk - 1) / blk;
        kernel_hier_scatter_sum<<<grid, blk, 0, s>>>(
            hc_mut->d_pred_scratch, hc_mut->d_dev_scratch,
            hc_mut->d_target_idx, hc_mut->d_coeff_scratch, nt);
    }

    // 7. Inverse Vilenkin (self-inverse)
    sp_cuda_vilenkin_inplace(hc_mut->d_coeff_scratch, pd, 1,
                              hc_mut->d_vilenkin_factors, hc_mut->n_factors, s);

    // 8. Unpad → output
    kernel_sqfree_unpad<<<1, 256, 0, s>>>(
        hc_mut->d_coeff_scratch, d_vec_out, hd, pd, 1);
}

// ── Public write/read API ─────────────────────────────────────────────

void sp_cuda_hier_write_k(sp_cuda_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *d_k_vec) {
    sp_cuda_hier_write_one(hc, layer, head, pos, d_k_vec,
                            hc->d_k_cache, hc->bytes_per_pos_k);
}

void sp_cuda_hier_write_v(sp_cuda_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *d_v_vec) {
    sp_cuda_hier_write_one(hc, layer, head, pos, d_v_vec,
                            hc->d_v_cache, hc->bytes_per_pos_v);
}

void sp_cuda_hier_read_k(const sp_cuda_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *d_k_out) {
    sp_cuda_hier_read_one(hc, layer, head, pos, d_k_out,
                           hc->d_k_cache, hc->bytes_per_pos_k);
}

void sp_cuda_hier_read_v(const sp_cuda_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *d_v_out) {
    sp_cuda_hier_read_one(hc, layer, head, pos, d_v_out,
                           hc->d_v_cache, hc->bytes_per_pos_v);
}

}  // extern "C"
