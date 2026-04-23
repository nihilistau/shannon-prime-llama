// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// CUDA kernels for the sqfree + spinor aggressive compression path.
//
// Additive to backends/cuda/shannon_prime_cuda.cu — the existing VHT2-at-p=2
// butterfly kernels are untouched. These new kernels handle:
//
//   1. Prime-Hartley (Vilenkin) transform — successive stages per prime factor
//   2. Möbius CSR prediction — segment-sum over skeleton values
//   3. Spinor sheet bit — pick min(|v_plus|, |v_minus|), store 1-bit flag
//   4. Sqfree pad/unpad — mean-fill padding for non-power-of-2 dimensions
//
// All kernels operate on device memory. No host↔device transfers needed
// when the shadow cache lives on GPU (same as the ship CUDA path).

#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Vilenkin prime-Hartley kernel
// ============================================================================
//
// One block per vector. Shared memory for the coefficients.
// Successive stages: for each prime factor p of N, apply the p×p Hartley
// matrix at the current stride. Self-inverse (normalized by 1/√p per stage).
//
// Max pad_dim = 330 (hd=256 → 330 = 2·3·5·11). Fits in 48KB shared memory.

__device__ float cas_val(float angle) {
    return __cosf(angle) + __sinf(angle);
}

__global__ void kernel_vilenkin_inplace(
    float *data,        // [n_vecs × pad_dim]
    int    pad_dim,     // Product of prime factors
    int    n_vecs,
    int    n_factors,
    int   *factors      // Device array of prime factors
) {
    extern __shared__ float smem[];

    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    float *vec = data + vec_idx * pad_dim;

    // Load into shared memory
    if (tid < pad_dim) {
        smem[tid] = vec[tid];
    }
    __syncthreads();

    // Successive Hartley stages
    int stride = 1;
    for (int f = 0; f < n_factors; f++) {
        int p = factors[f];
        int block_size = stride * p;
        int n_blocks_inner = pad_dim / block_size;
        float inv_sqrt_p = rsqrtf((float)p);

        // Each thread handles one (block, stride_pos) pair
        int work_items = n_blocks_inner * stride;
        for (int wi = tid; wi < work_items; wi += blockDim.x) {
            int blk = wi / stride;
            int s   = wi % stride;
            int base = blk * block_size + s;

            // Gather p elements
            float gathered[16]; // Max prime = 13
            for (int k = 0; k < p; k++) {
                gathered[k] = smem[base + k * stride];
            }

            // Apply p×p Hartley matrix
            float result[16];
            for (int k = 0; k < p; k++) {
                float sum = 0.0f;
                for (int j = 0; j < p; j++) {
                    float angle = 2.0f * (float)M_PI * (float)(k * j) / (float)p;
                    sum += cas_val(angle) * gathered[j];
                }
                result[k] = sum * inv_sqrt_p;
            }

            // Scatter back
            for (int k = 0; k < p; k++) {
                smem[base + k * stride] = result[k];
            }
        }

        stride *= p;
        __syncthreads();
    }

    // Write back
    if (tid < pad_dim) {
        vec[tid] = smem[tid];
    }
}

// ============================================================================
// Sqfree pad/unpad kernels
// ============================================================================

__global__ void kernel_sqfree_pad(
    const float *in,    // [n_vecs × head_dim]
    float       *out,   // [n_vecs × pad_dim]
    int          head_dim,
    int          pad_dim,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    const float *src = in  + vec_idx * head_dim;
    float       *dst = out + vec_idx * pad_dim;

    // Compute mean for padding (warp reduction)
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        sum += src[i];
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / (float)head_dim;
    }
    __syncthreads();

    // Copy data + pad
    for (int i = tid; i < pad_dim; i += blockDim.x) {
        dst[i] = (i < head_dim) ? src[i] : s_mean;
    }
}

__global__ void kernel_sqfree_unpad(
    const float *in,    // [n_vecs × pad_dim]
    float       *out,   // [n_vecs × head_dim]
    int          head_dim,
    int          pad_dim,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int tid     = threadIdx.x;
    if (vec_idx >= n_vecs) return;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        out[vec_idx * head_dim + i] = in[vec_idx * pad_dim + i];
    }
}

// ============================================================================
// Möbius CSR prediction kernel
// ============================================================================
//
// For each residual position i, compute:
//   pred[i] = Σ csr_mu_sign[j] · skel_vals[csr_skel_slot[j]]
//             for j in [csr_offsets[i], csr_offsets[i+1])
//
// One thread per residual position. CSR tables are device-resident constants.

__global__ void kernel_mobius_predict(
    const float *skel_vals,    // [n_vecs × sk_k]
    float       *pred_out,     // [n_vecs × n_res]
    const int   *csr_offsets,  // [n_res + 1]
    const int   *csr_skel_slot,// [n_terms]
    const int   *csr_mu_sign,  // [n_terms] — ±1 as int
    int          sk_k,
    int          n_res,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int res_i   = threadIdx.x;
    if (vec_idx >= n_vecs || res_i >= n_res) return;

    const float *sv = skel_vals + vec_idx * sk_k;
    int start = csr_offsets[res_i];
    int end   = csr_offsets[res_i + 1];

    float pred = 0.0f;
    for (int j = start; j < end; j++) {
        pred += (float)csr_mu_sign[j] * sv[csr_skel_slot[j]];
    }

    pred_out[vec_idx * n_res + res_i] = pred;
}

// ============================================================================
// Spinor sheet bit kernel (compress path)
// ============================================================================
//
// For each residual position, compare |actual - pred| vs |actual + pred|.
// Store the sheet bit (1 if flipped) and the winning deviation.

__global__ void kernel_spinor_extract(
    const float *actual_res,   // [n_vecs × n_res] — actual coeff at residual positions
    const float *pred,         // [n_vecs × n_res] — Möbius predictions
    float       *deviation,    // [n_vecs × n_res] — output: winning deviation
    uint8_t     *sheet_packed, // [n_vecs × (n_res+7)/8] — output: packed sheet bits
    int          n_res,
    int          n_vecs
) {
    int vec_idx = blockIdx.x;
    int res_i   = threadIdx.x;
    if (vec_idx >= n_vecs || res_i >= n_res) return;

    int idx = vec_idx * n_res + res_i;
    float a = actual_res[idx];
    float p = pred[idx];

    float v_plus  = a - p;
    float v_minus = a + p;

    bool use_minus = fabsf(v_minus) < fabsf(v_plus);
    deviation[idx] = use_minus ? v_minus : v_plus;

    // Pack sheet bit (atomicOr for thread safety within a byte)
    if (use_minus) {
        int byte_idx = vec_idx * ((n_res + 7) / 8) + res_i / 8;
        atomicOr((unsigned int *)(sheet_packed + (byte_idx & ~3)),
                 (1u << (res_i % 8)) << ((byte_idx & 3) * 8));
    }
}

// ============================================================================
// Spinor correction kernel (reconstruct path)
// ============================================================================
//
// Flip pred sign where sheet bit is set, add dequantized residual,
// scatter to coefficient vector.

__global__ void kernel_spinor_reconstruct(
    float       *coeffs,       // [n_vecs × pad_dim] — output coefficient vector
    const float *skel_vals,    // [n_vecs × sk_k] — (unused; kept for signature
                               //   parity with a possible fused variant)
    const float *deviation,    // [n_vecs × n_res] — dequantized residual
    const float *pred,         // [n_vecs × n_res] — Möbius predictions (recomputed)
    const uint8_t *sheet_packed,// [n_vecs × (n_res+7)/8]
    const int   *residual_idx, // [n_res] — where in coeffs[] to write
    int          pad_dim,
    int          sk_k,
    int          n_res,
    int          n_vecs,
    int          use_spinor
) {
    (void)skel_vals; (void)sk_k;
    int vec_idx = blockIdx.x;
    if (vec_idx >= n_vecs) return;

    // Strided loop so blockDim.x < n_res still covers every residual position
    // (matches kernel_reconstruct_residual_batch's convention).
    int sheet_stride = (n_res + 7) / 8;
    for (int res_i = threadIdx.x; res_i < n_res; res_i += blockDim.x) {
        float p = pred[vec_idx * n_res + res_i];

        if (use_spinor) {
            uint8_t byte_val = sheet_packed[vec_idx * sheet_stride + res_i / 8];
            if (byte_val & (1u << (res_i % 8))) {
                p = -p;
            }
        }

        float val = p + deviation[vec_idx * n_res + res_i];
        coeffs[vec_idx * pad_dim + residual_idx[res_i]] = val;
    }
}

// ============================================================================
// Host-side dispatch helpers
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Launch Vilenkin transform on a batch of vectors
void sp_cuda_vilenkin_inplace(float *d_data, int pad_dim, int n_vecs,
                              int *d_factors, int n_factors,
                              cudaStream_t stream) {
    int smem_bytes = pad_dim * sizeof(float);
    int block_size = (pad_dim < 256) ? pad_dim : 256;
    kernel_vilenkin_inplace<<<n_vecs, block_size, smem_bytes, stream>>>(
        d_data, pad_dim, n_vecs, n_factors, d_factors
    );
}

// Launch sqfree pad
void sp_cuda_sqfree_pad(const float *d_in, float *d_out,
                        int head_dim, int pad_dim, int n_vecs,
                        cudaStream_t stream) {
    kernel_sqfree_pad<<<n_vecs, 256, 0, stream>>>(
        d_in, d_out, head_dim, pad_dim, n_vecs
    );
}

// Launch Möbius prediction
void sp_cuda_mobius_predict(const float *d_skel, float *d_pred,
                           const int *d_offsets, const int *d_slots,
                           const int *d_signs,
                           int sk_k, int n_res, int n_vecs,
                           cudaStream_t stream) {
    int block = (n_res < 256) ? n_res : 256;
    kernel_mobius_predict<<<n_vecs, block, 0, stream>>>(
        d_skel, d_pred, d_offsets, d_slots, d_signs,
        sk_k, n_res, n_vecs
    );
}

// Launch spinor extract (compress path)
void sp_cuda_spinor_extract(const float *d_actual, const float *d_pred,
                            float *d_deviation, uint8_t *d_sheet,
                            int n_res, int n_vecs,
                            cudaStream_t stream) {
    // Zero sheet bits first
    int sheet_bytes = n_vecs * ((n_res + 7) / 8);
    cudaMemsetAsync(d_sheet, 0, sheet_bytes, stream);

    int block = (n_res < 256) ? n_res : 256;
    kernel_spinor_extract<<<n_vecs, block, 0, stream>>>(
        d_actual, d_pred, d_deviation, d_sheet, n_res, n_vecs
    );
}

}  // extern "C" (closes the block opened near the top of this file)

// ============================================================================
// Step-3-MVP additions: kernels and cache wrapper for GPU-resident sqfree
// compress/decompress. No spinor support (deferred — see
// docs/STEP3-GPU-SQFREE-CACHE.md "Full scope").
// ============================================================================

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "shannon_prime_cuda.h"
#include "../../core/shannon_prime.h"

// ── helper kernels ────────────────────────────────────────────────

// Gather: out[i] = in[idx[i]] for i in [0, n)
__global__ void kernel_gather(const float *in, const int *idx,
                              float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[idx[i]];
}

// Scatter: out[idx[i]] = in[i]. Caller must zero `out` first if it needs
// untouched positions to be 0 (trivial via cudaMemsetAsync before launch).
__global__ void kernel_scatter(const float *in, const int *idx,
                               float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[idx[i]] = in[i];
}

// Batched scatter: each vec v scatters its own [sk_k] input into its own
// [pad_dim] slab of output. idx[] is shared across all vecs (same mask).
// One block per vec, threads cooperate across the sk_k indices.
__global__ void kernel_scatter_batch(const float *in_base,  int in_stride,
                                     const int *idx,
                                     float *out_base,        int out_stride,
                                     int sk_k,               int n_vecs)
{
    int v = blockIdx.x;
    if (v >= n_vecs) return;
    const float *in  = in_base  + (size_t)v * in_stride;
    float       *out = out_base + (size_t)v * out_stride;
    for (int k = threadIdx.x; k < sk_k; k += blockDim.x) {
        out[idx[k]] = in[k];
    }
}

// Batched reconstruct: coeff[residual_idx[r]] = pred[r] + dev[r] per vec.
// in_base/dev_base are [n_vecs * n_res], coeff_base is [n_vecs * pad_dim].
__global__ void kernel_reconstruct_residual_batch(
    const float *pred_base, const float *dev_base,
    const int *residual_idx, float *coeff_base,
    int n_res, int pd, int n_vecs)
{
    int v = blockIdx.x;
    if (v >= n_vecs) return;
    const float *pred  = pred_base  + (size_t)v * n_res;
    const float *dev   = dev_base   + (size_t)v * n_res;
    float       *coeff = coeff_base + (size_t)v * pd;
    for (int r = threadIdx.x; r < n_res; r += blockDim.x) {
        coeff[residual_idx[r]] = pred[r] + dev[r];
    }
}

// Batched dequantize: per-vec mag drives the step size. levels_base is
// [n_vecs * n_res], d_mag_base is [n_vecs], dev_base is [n_vecs * n_res].
__global__ void kernel_dequantize_residual_batch(
    const uint8_t *levels_base, int n_res, int bits,
    const float *d_mag_base, float *dev_base, int n_vecs)
{
    int v = blockIdx.x;
    if (v >= n_vecs) return;
    int   L      = 1 << bits;
    float center = 0.5f * (float)(L - 1);
    float mag    = d_mag_base[v];
    float sat    = mag * (float)bits;
    float step   = (L > 1) ? (2.0f * sat / (float)(L - 1)) : 0.0f;
    const uint8_t *levels = levels_base + (size_t)v * n_res;
    float         *dev    = dev_base    + (size_t)v * n_res;
    for (int r = threadIdx.x; r < n_res; r += blockDim.x) {
        dev[r] = (((float)levels[r]) - center) * step;
    }
}

// Batched bit-unpack: each vec's packed bytes at a strided offset into
// the cache slot. One block per vec (so indexing is sequential within
// a vec's bit stream), threads within the block cooperate over n_res
// residual positions.
__global__ void kernel_unpack_levels_batch(
    const uint8_t *cache_base, int bytes_per_pos,
    int offset_in_slot, int n_res, int bits,
    uint8_t *levels_base, int packed_bytes, int n_vecs)
{
    int v = blockIdx.x;
    if (v >= n_vecs) return;
    const uint8_t *packed = cache_base + (size_t)v * bytes_per_pos + offset_in_slot;
    uint8_t       *out    = levels_base + (size_t)v * n_res;
    const uint32_t mask_bits = (1u << bits) - 1u;
    for (int r = threadIdx.x; r < n_res; r += blockDim.x) {
        int bit_off = r * bits;
        uint32_t w = (uint32_t)packed[bit_off / 8] >> (bit_off % 8);
        if ((bit_off % 8) + bits > 8 && (bit_off / 8 + 1) < packed_bytes) {
            w |= ((uint32_t)packed[bit_off / 8 + 1]) << (8 - (bit_off % 8));
        }
        out[r] = (uint8_t)(w & mask_bits);
    }
}

// Gather mag fp32 from each slot's mag offset into a contiguous [n_vecs] array.
__global__ void kernel_gather_mag_batch(
    const uint8_t *cache_base, int bytes_per_pos,
    int mag_offset_in_slot, float *d_mag_out, int n_vecs)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vecs) return;
    const uint8_t *src = cache_base + (size_t)v * bytes_per_pos + mag_offset_in_slot;
    // mag is 4 aligned bytes — memcpy preserves alignment semantics
    float m;
    memcpy(&m, src, 4);
    d_mag_out[v] = m;
}

// Fused: dev[i] = coeff[residual_idx[i]] - pred[i]. MVP (no spinor).
// Used only on the write path.
__global__ void kernel_residual_deviation_mvp(
    const float *coeff, const int *residual_idx,
    const float *pred, float *dev, int n_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_res) {
        float actual = coeff[residual_idx[i]];
        dev[i] = actual - pred[i];
    }
}

// Read-path reconstruction: coeff[residual_idx[i]] = pred[i] + dev[i].
// Used on the read path to scatter reconstructed residual positions
// back into the coefficient vector. MVP (no spinor).
__global__ void kernel_reconstruct_residual_mvp(
    const float *pred, const float *dev,
    const int *residual_idx, float *coeff, int n_res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_res) {
        coeff[residual_idx[i]] = pred[i] + dev[i];
    }
}

// Single-block reduction: mag = mean(|dev[i]|) over n_res elements.
// One block, up to 256 threads. Writes mag (single fp32) to d_mag.
__global__ void kernel_mean_abs(const float *dev, int n_res, float *d_mag) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n_res; i += blockDim.x) local += fabsf(dev[i]);
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float mag = smem[0] / (float)n_res;
        if (mag < 1e-12f) mag = 1e-12f;
        *d_mag = mag;
    }
}

// Residual quantize — MUST match sp_quantize_residual in
// core/shannon_prime_sqfree.c exactly. Saturation is `mag * bits`,
// step = 2*sat/(L-1), level = round-half-up(val/step + center).
__global__ void kernel_quantize_residual(
    const float *dev, int n_res, int bits, const float *d_mag,
    uint8_t *levels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_res) {
        int   L        = 1 << bits;
        float mag      = *d_mag;
        float center   = 0.5f * (float)(L - 1);
        float sat      = (mag > 1e-12f) ? (mag * (float)bits) : 1e-12f;
        float step     = (2.0f * sat) / (float)(L - 1);
        float inv_step = 1.0f / step;
        float q_f      = dev[i] * inv_step + center;
        int   level    = (int)(q_f + 0.5f);   // matches CPU exactly
        if (level < 0)     level = 0;
        if (level >= L)    level = L - 1;
        levels[i] = (uint8_t)level;
    }
}

// Residual dequantize — mirrors sp_dequantize_residual exactly.
__global__ void kernel_dequantize_residual(
    const uint8_t *levels, int n_res, int bits, const float *d_mag,
    float *dev)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_res) {
        int   L      = 1 << bits;
        float mag    = *d_mag;
        float center = 0.5f * (float)(L - 1);
        float sat    = mag * (float)bits;
        float step   = (L > 1) ? (2.0f * sat / (float)(L - 1)) : 0.0f;
        dev[i] = (((float)levels[i]) - center) * step;
    }
}

// Level packing (LSB-first, matches sp_sqfree_compress_one layout).
// Single thread per block — cheap serialization, matches CPU exactly.
__global__ void kernel_pack_levels(const uint8_t *levels, int n_res,
                                    int bits, uint8_t *packed,
                                    int packed_bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < packed_bytes; ++i) packed[i] = 0;
    for (int i = 0; i < n_res; ++i) {
        int bit_off = i * bits;
        packed[bit_off / 8] |= (uint8_t)(levels[i] << (bit_off % 8));
        if ((bit_off % 8) + bits > 8 && (bit_off / 8 + 1) < packed_bytes) {
            packed[bit_off / 8 + 1] |= (uint8_t)(levels[i] >> (8 - (bit_off % 8)));
        }
    }
}

__global__ void kernel_unpack_levels(const uint8_t *packed, int n_res,
                                      int bits, uint8_t *levels,
                                      int packed_bytes) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t mask_bits = (1u << bits) - 1u;
    for (int i = 0; i < n_res; ++i) {
        int bit_off = i * bits;
        uint32_t w = (uint32_t)packed[bit_off / 8] >> (bit_off % 8);
        if ((bit_off % 8) + bits > 8 && (bit_off / 8 + 1) < packed_bytes) {
            w |= ((uint32_t)packed[bit_off / 8 + 1]) << (8 - (bit_off % 8));
        }
        levels[i] = (uint8_t)(w & mask_bits);
    }
}

// ── sp_cuda_sqfree_cache_t wrapper ────────────────────────────────

extern "C" {

int sp_cuda_sqfree_cache_init(sp_cuda_sqfree_cache_t *cc,
                               const sp_config_t *cfg,
                               int max_seq_len,
                               int residual_bits,
                               int use_spinor,
                               void *stream)
{
    memset(cc, 0, sizeof(*cc));
    memcpy(&cc->config, cfg, sizeof(sp_config_t));
    cc->max_seq_len   = max_seq_len;
    cc->residual_bits = residual_bits;
    cc->use_spinor    = use_spinor;      // MVP path passes 0; stored for layout
    cc->stream        = stream;

    // Pad dimension is derived from head_dim.
    int pad_dim = sp_sqfree_pad_dim(cfg->head_dim);
    cc->pad_dim = pad_dim;

    // Build Knight mask on host (L/2 skeleton, squarefree-first when no
    // calibration variance is available yet).
    sp_knight_mask_t host_mask;
    if (sp_knight_mask_init(&host_mask, pad_dim, pad_dim / 2, NULL) != 0) {
        fprintf(stderr, "[sp-cuda-sqfree] knight_mask_init failed\n");
        return -1;
    }
    cc->sk_k     = host_mask.sk_k;
    cc->n_res    = host_mask.n_res;
    cc->n_terms  = host_mask.n_terms;

    // Band configs operate on the skeleton (sk_k), not pad_dim.
    int default_k_bits[5] = {5, 4, 4, 4, 5};
    int default_v_bits[5] = {5, 4, 4, 4, 5};
    int k_nb = (cfg->k_n_bands > 0 && cfg->k_n_bands <= SP_MAX_BANDS) ? cfg->k_n_bands : 5;
    int v_nb = (cfg->v_n_bands > 0 && cfg->v_n_bands <= SP_MAX_BANDS) ? cfg->v_n_bands : 5;
    int k_bits[SP_MAX_BANDS];
    int v_bits[SP_MAX_BANDS];
    for (int i = 0; i < k_nb; ++i) k_bits[i] = (cfg->k_band_bits[i] > 0) ? cfg->k_band_bits[i] : default_k_bits[i % 5];
    for (int i = 0; i < v_nb; ++i) v_bits[i] = (cfg->v_band_bits[i] > 0) ? cfg->v_band_bits[i] : default_v_bits[i % 5];
    sp_band_config_init(&cc->k_bands, cc->sk_k, k_nb, k_bits);
    sp_band_config_init(&cc->v_bands, cc->sk_k, v_nb, v_bits);

    // Upload mask arrays.
    cudaMalloc(&cc->d_skeleton_idx, (size_t)cc->sk_k  * sizeof(int));
    cudaMalloc(&cc->d_residual_idx, (size_t)cc->n_res * sizeof(int));
    cudaMalloc(&cc->d_csr_offsets,  (size_t)(cc->n_res + 1) * sizeof(int));
    cudaMalloc(&cc->d_csr_skel_slot,(size_t)cc->n_terms * sizeof(int));
    // The CUDA mobius_predict kernel takes int32 signs (see
    // kernel_mobius_predict above). Convert from the CPU's int8_t
    // representation to int32 before upload.
    cudaMalloc(&cc->d_csr_mu_sign,  (size_t)cc->n_terms * sizeof(int));
    cudaMemcpy(cc->d_skeleton_idx, host_mask.skeleton_idx,
               (size_t)cc->sk_k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cc->d_residual_idx, host_mask.residual_idx,
               (size_t)cc->n_res * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cc->d_csr_offsets, host_mask.csr_offsets,
               (size_t)(cc->n_res + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cc->d_csr_skel_slot, host_mask.csr_skel_slot,
               (size_t)cc->n_terms * sizeof(int), cudaMemcpyHostToDevice);
    {
        int *signs_i32 = (int *)malloc((size_t)cc->n_terms * sizeof(int));
        if (!signs_i32) {
            fprintf(stderr, "[sp-cuda-sqfree] signs malloc failed\n");
            sp_knight_mask_free(&host_mask);
            return -1;
        }
        for (int i = 0; i < cc->n_terms; ++i) {
            signs_i32[i] = (int)host_mask.csr_mu_sign[i];
        }
        cudaMemcpy(cc->d_csr_mu_sign, signs_i32,
                   (size_t)cc->n_terms * sizeof(int), cudaMemcpyHostToDevice);
        free(signs_i32);
    }
    sp_knight_mask_free(&host_mask);

    // Vilenkin factor list for the staged kernel.
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
            fprintf(stderr, "[sp-cuda-sqfree] pad_dim %d doesn't factor over "
                            "{2,3,5,7,11}; cache init failed\n", pad_dim);
            return -1;
        }
    }
    cc->n_factors = nf;
    cudaMalloc(&cc->d_vilenkin_factors, (size_t)nf * sizeof(int));
    cudaMemcpy(cc->d_vilenkin_factors, factors,
               (size_t)nf * sizeof(int), cudaMemcpyHostToDevice);

    // Compressed storage layout.
    int res_bytes   = (cc->n_res * cc->residual_bits + 7) / 8;
    int sheet_bytes = use_spinor ? ((cc->n_res + 7) / 8) : 0;
    cc->bytes_per_pos_k = cc->k_bands.total_bytes + 4 + res_bytes + sheet_bytes;
    cc->bytes_per_pos_v = cc->v_bands.total_bytes + 4 + res_bytes + sheet_bytes;
    cc->n_slots = cfg->n_layers * cfg->n_heads_kv;

    size_t k_total = (size_t)cc->n_slots * max_seq_len * cc->bytes_per_pos_k;
    size_t v_total = (size_t)cc->n_slots * max_seq_len * cc->bytes_per_pos_v;
    cudaMalloc(&cc->d_k_cache, k_total);
    cudaMalloc(&cc->d_v_cache, v_total);
    cudaMemsetAsync(cc->d_k_cache, 0, k_total, (cudaStream_t)stream);
    cudaMemsetAsync(cc->d_v_cache, 0, v_total, (cudaStream_t)stream);

    // Scratch buffers.
    // Scratch sized for batched reads: the batch path processes up to
    // max_seq_len positions in one pass so each per-vec scratch needs
    // n_pos slots. Total ~1.5 MB for pad_dim=154, max_seq=1032 — negligible.
    cudaMalloc(&cc->d_pad_scratch,   (size_t)pad_dim * max_seq_len * sizeof(float));
    cudaMalloc(&cc->d_coeff_scratch, (size_t)pad_dim * max_seq_len * sizeof(float));
    cudaMalloc(&cc->d_skel_scratch,  (size_t)cc->sk_k * max_seq_len * sizeof(float));
    cudaMalloc(&cc->d_pred_scratch,  (size_t)cc->n_res * max_seq_len * sizeof(float));
    cudaMalloc(&cc->d_dev_scratch,   (size_t)cc->n_res * max_seq_len * sizeof(float));
    cudaMalloc(&cc->d_levels_scratch,(size_t)cc->n_res * max_seq_len);
    cudaMalloc(&cc->d_mag_scratch,   (size_t)max_seq_len * sizeof(float));
    // Spinor scratches (always allocated; kernel paths gate on use_spinor).
    // Sized for n_pos up to max_seq_len to cover batch read reconstruction.
    {
        int sheet_bytes_per_vec = (cc->n_res + 7) / 8;
        cudaMalloc(&cc->d_actual_scratch,
                   (size_t)cc->n_res * max_seq_len * sizeof(float));
        cudaMalloc(&cc->d_sheet_scratch,
                   (size_t)sheet_bytes_per_vec * max_seq_len);
    }

    fprintf(stderr,
        "[sp-cuda-sqfree] init: hd=%d pad_dim=%d sk_k=%d n_res=%d n_terms=%d "
        "res_bits=%d spinor=%d bytes_per_pos k=%d v=%d total=%.2f MB\n",
        cfg->head_dim, pad_dim, cc->sk_k, cc->n_res, cc->n_terms,
        residual_bits, use_spinor, cc->bytes_per_pos_k, cc->bytes_per_pos_v,
        (k_total + v_total) / (1024.0 * 1024.0));
    return 0;
}

void sp_cuda_sqfree_cache_free(sp_cuda_sqfree_cache_t *cc) {
    if (!cc) return;
    if (cc->d_skeleton_idx)    cudaFree(cc->d_skeleton_idx);
    if (cc->d_residual_idx)    cudaFree(cc->d_residual_idx);
    if (cc->d_csr_offsets)     cudaFree(cc->d_csr_offsets);
    if (cc->d_csr_skel_slot)   cudaFree(cc->d_csr_skel_slot);
    if (cc->d_csr_mu_sign)     cudaFree(cc->d_csr_mu_sign);
    if (cc->d_vilenkin_factors)cudaFree(cc->d_vilenkin_factors);
    if (cc->d_k_cache)         cudaFree(cc->d_k_cache);
    if (cc->d_v_cache)         cudaFree(cc->d_v_cache);
    if (cc->d_pad_scratch)     cudaFree(cc->d_pad_scratch);
    if (cc->d_coeff_scratch)   cudaFree(cc->d_coeff_scratch);
    if (cc->d_skel_scratch)    cudaFree(cc->d_skel_scratch);
    if (cc->d_pred_scratch)    cudaFree(cc->d_pred_scratch);
    if (cc->d_dev_scratch)     cudaFree(cc->d_dev_scratch);
    if (cc->d_levels_scratch)  cudaFree(cc->d_levels_scratch);
    if (cc->d_mag_scratch)     cudaFree(cc->d_mag_scratch);
    if (cc->d_actual_scratch)  cudaFree(cc->d_actual_scratch);
    if (cc->d_sheet_scratch)   cudaFree(cc->d_sheet_scratch);
    memset(cc, 0, sizeof(*cc));
}

// Internal write: compresses one (layer, head, pos) into dest slot using
// band config `bc`. Runs entirely on the cache's CUDA stream.
static void sp_cuda_sqfree_write_one(sp_cuda_sqfree_cache_t *cc,
                                      int layer, int head, int pos,
                                      const float *d_vec,
                                      const sp_band_config_t *bc,
                                      uint8_t *d_cache,
                                      int bytes_per_pos)
{
    const int hd      = cc->config.head_dim;
    const int pd      = cc->pad_dim;
    const int sk_k    = cc->sk_k;
    const int n_res   = cc->n_res;
    cudaStream_t s    = (cudaStream_t)cc->stream;

    // 1. Sqfree pad → d_pad_scratch
    kernel_sqfree_pad<<<1, 256, 0, s>>>(d_vec, cc->d_pad_scratch, hd, pd, 1);

    // 2. Vilenkin forward on pad_scratch
    cudaMemcpyAsync(cc->d_coeff_scratch, cc->d_pad_scratch,
                    (size_t)pd * sizeof(float),
                    cudaMemcpyDeviceToDevice, s);
    sp_cuda_vilenkin_inplace(cc->d_coeff_scratch, pd, 1,
                              cc->d_vilenkin_factors, cc->n_factors, s);

    // 3. Gather skeleton → d_skel_scratch
    {
        int blk = 128;
        int grid = (sk_k + blk - 1) / blk;
        kernel_gather<<<grid, blk, 0, s>>>(
            cc->d_coeff_scratch, cc->d_skeleton_idx,
            cc->d_skel_scratch, sk_k);
    }

    // 4. Band-quantize skeleton directly into the slot head.
    int slot = layer * cc->config.n_heads_kv + head;
    uint8_t *dest = d_cache + (size_t)slot * cc->max_seq_len * bytes_per_pos
                           + (size_t)pos * bytes_per_pos;
    sp_cuda_band_quantize(cc->d_skel_scratch, dest, bc, 1, s);

    if (n_res == 0) return;

    // 5. Möbius predict residual
    sp_cuda_mobius_predict(cc->d_skel_scratch, cc->d_pred_scratch,
                            cc->d_csr_offsets, cc->d_csr_skel_slot,
                            cc->d_csr_mu_sign,
                            sk_k, n_res, 1, s);

    // 6. Deviation.
    //   No-spinor path: dev[i] = coeff[residual_idx[i]] - pred[i]
    //   Spinor path   : gather actual = coeff[residual_idx[:]], then
    //                   dev[i] = (|a-p| <= |a+p|) ? a-p : a+p
    //                   and pack the sheet-choice bit per residual.
    if (cc->use_spinor) {
        // 6a. Gather actual residual coefficients into d_actual_scratch.
        {
            int blk = 128;
            int grid = (n_res + blk - 1) / blk;
            kernel_gather<<<grid, blk, 0, s>>>(
                cc->d_coeff_scratch, cc->d_residual_idx,
                cc->d_actual_scratch, n_res);
        }
        // 6b. spinor_extract writes winning deviation + packed sheet bits.
        sp_cuda_spinor_extract(cc->d_actual_scratch, cc->d_pred_scratch,
                                cc->d_dev_scratch, cc->d_sheet_scratch,
                                n_res, 1, s);
    } else {
        int blk = 128;
        int grid = (n_res + blk - 1) / blk;
        kernel_residual_deviation_mvp<<<grid, blk, 0, s>>>(
            cc->d_coeff_scratch, cc->d_residual_idx,
            cc->d_pred_scratch, cc->d_dev_scratch, n_res);
    }

    // 7. mag = mean(|dev|)
    kernel_mean_abs<<<1, 256, 0, s>>>(cc->d_dev_scratch, n_res,
                                       cc->d_mag_scratch);
    // Copy mag fp32 into the slot right after skeleton bytes.
    cudaMemcpyAsync(dest + bc->total_bytes, cc->d_mag_scratch, 4,
                    cudaMemcpyDeviceToDevice, s);

    // 8. Quantize residual into levels
    {
        int blk = 128;
        int grid = (n_res + blk - 1) / blk;
        kernel_quantize_residual<<<grid, blk, 0, s>>>(
            cc->d_dev_scratch, n_res, cc->residual_bits,
            cc->d_mag_scratch, cc->d_levels_scratch);
    }

    // 9. Pack levels into the slot (LSB-first matches CPU layout)
    int res_bytes = (n_res * cc->residual_bits + 7) / 8;
    kernel_pack_levels<<<1, 1, 0, s>>>(
        cc->d_levels_scratch, n_res, cc->residual_bits,
        dest + bc->total_bytes + 4, res_bytes);

    // 10. Spinor sheet bytes sit immediately after the packed residual in
    //     the slot (see cache_init: sheet_bytes = (n_res+7)/8 when
    //     use_spinor=1, and bytes_per_pos reserves that tail region).
    if (cc->use_spinor) {
        int sheet_bytes = (n_res + 7) / 8;
        cudaMemcpyAsync(dest + bc->total_bytes + 4 + res_bytes,
                        cc->d_sheet_scratch, (size_t)sheet_bytes,
                        cudaMemcpyDeviceToDevice, s);
    }
}

// Internal read: decompresses one (layer, head, pos) slot into d_vec_out
// using band config `bc`.
static void sp_cuda_sqfree_read_one(const sp_cuda_sqfree_cache_t *cc,
                                     int layer, int head, int pos,
                                     float *d_vec_out,
                                     const sp_band_config_t *bc,
                                     const uint8_t *d_cache,
                                     int bytes_per_pos)
{
    const int hd    = cc->config.head_dim;
    const int pd    = cc->pad_dim;
    const int sk_k  = cc->sk_k;
    const int n_res = cc->n_res;
    cudaStream_t s  = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    const uint8_t *src = d_cache + (size_t)slot * cc->max_seq_len * bytes_per_pos
                                 + (size_t)pos * bytes_per_pos;

    // 1. Dequantize skeleton → d_skel_scratch
    sp_cuda_band_dequantize(src, cc->d_skel_scratch, bc, 1, s);

    // 2. Zero coeff scratch + scatter skeleton back in
    cudaMemsetAsync(cc->d_coeff_scratch, 0, (size_t)pd * sizeof(float), s);
    {
        int blk = 128;
        int grid = (sk_k + blk - 1) / blk;
        kernel_scatter<<<grid, blk, 0, s>>>(
            cc->d_skel_scratch, cc->d_skeleton_idx,
            cc->d_coeff_scratch, sk_k);
    }

    if (n_res > 0) {
        // 3. Möbius predict
        sp_cuda_mobius_predict(cc->d_skel_scratch,
                                (float *)cc->d_pred_scratch,
                                cc->d_csr_offsets, cc->d_csr_skel_slot,
                                cc->d_csr_mu_sign,
                                sk_k, n_res, 1, s);

        // 4. Read mag from slot
        cudaMemcpyAsync((void *)cc->d_mag_scratch,
                        src + bc->total_bytes, 4,
                        cudaMemcpyDeviceToDevice, s);

        // 5. Unpack levels
        int res_bytes = (n_res * cc->residual_bits + 7) / 8;
        kernel_unpack_levels<<<1, 1, 0, s>>>(
            src + bc->total_bytes + 4, n_res, cc->residual_bits,
            (uint8_t *)cc->d_levels_scratch, res_bytes);

        // 6. Dequantize levels → deviation
        {
            int blk = 128;
            int grid = (n_res + blk - 1) / blk;
            kernel_dequantize_residual<<<grid, blk, 0, s>>>(
                cc->d_levels_scratch, n_res, cc->residual_bits,
                cc->d_mag_scratch, (float *)cc->d_dev_scratch);
        }

        // 7. Reconstruct residual positions.
        //   No-spinor: coeff[residual_idx[i]] = pred[i] + dev[i]
        //   Spinor   : load sheet bits from slot, flip pred sign where
        //              bit set, then scatter pred+dev into coeff.
        if (cc->use_spinor) {
            int sheet_bytes = (n_res + 7) / 8;
            cudaMemcpyAsync(cc->d_sheet_scratch,
                            src + bc->total_bytes + 4 + res_bytes,
                            (size_t)sheet_bytes,
                            cudaMemcpyDeviceToDevice, s);
            int block = (n_res < 256) ? n_res : 256;
            kernel_spinor_reconstruct<<<1, block, 0, s>>>(
                cc->d_coeff_scratch,
                /*skel_vals*/ NULL,
                (const float *)cc->d_dev_scratch,
                cc->d_pred_scratch,
                cc->d_sheet_scratch,
                cc->d_residual_idx,
                pd, sk_k, n_res, /*n_vecs*/ 1, /*use_spinor*/ 1);
        } else {
            int blk = 128;
            int grid = (n_res + blk - 1) / blk;
            kernel_reconstruct_residual_mvp<<<grid, blk, 0, s>>>(
                cc->d_pred_scratch, (const float *)cc->d_dev_scratch,
                cc->d_residual_idx, cc->d_coeff_scratch, n_res);
        }
    }

    // 8. Vilenkin inverse (= forward; self-inverse)
    sp_cuda_vilenkin_inplace(cc->d_coeff_scratch, pd, 1,
                              cc->d_vilenkin_factors, cc->n_factors, s);

    // 9. Sqfree unpad → d_vec_out
    kernel_sqfree_unpad<<<1, 256, 0, s>>>(cc->d_coeff_scratch,
                                           d_vec_out, hd, pd, 1);
}

// Internal batched read: reads n_pos contiguous positions for one
// (layer, head) in a single-per-step kernel dispatch series.
// Same output format as per-vec read but the entire range is
// processed in parallel — cuts kernel launch count from ~9*n_pos
// to ~9 total, which is the step-3 kernel-launch-overhead fix.
static void sp_cuda_sqfree_read_batch_one(
    const sp_cuda_sqfree_cache_t *cc,
    int layer, int head, int start_pos, int n_pos,
    float *d_out,                    // [n_pos * head_dim] contiguous
    const sp_band_config_t *bc,
    const uint8_t *d_cache, int bytes_per_pos)
{
    if (n_pos <= 0) return;
    const int hd    = cc->config.head_dim;
    const int pd    = cc->pad_dim;
    const int sk_k  = cc->sk_k;
    const int n_res = cc->n_res;
    cudaStream_t s  = (cudaStream_t)cc->stream;

    int slot = layer * cc->config.n_heads_kv + head;
    const uint8_t *src_base = d_cache
                            + (size_t)slot * cc->max_seq_len * bytes_per_pos
                            + (size_t)start_pos * bytes_per_pos;

    // 1. Repack skeleton bytes.
    //
    // Each slot's layout is [total_bytes skeleton][4 mag][res_bytes residual]
    // so stride between vecs = bytes_per_pos, not total_bytes — which is
    // what sp_cuda_band_dequantize's n_vecs loop assumes. Repack the
    // contiguous [n_pos][total_bytes] view into d_pad_scratch (reused as
    // a byte buffer, unused at this stage of the pipeline) via a strided
    // memcpy, then run the existing batched dequantize kernel.
    uint8_t *d_skel_bytes = (uint8_t *)cc->d_pad_scratch;
    cudaMemcpy2DAsync(d_skel_bytes,    (size_t)bc->total_bytes,
                      src_base,        (size_t)bytes_per_pos,
                      (size_t)bc->total_bytes, (size_t)n_pos,
                      cudaMemcpyDeviceToDevice, s);

    // Dequantize all n_pos skeletons in one batch (existing kernel
    // already takes n_vecs). Input: contiguous [n_pos * total_bytes];
    // output: [n_pos * sk_k] contiguous fp32.
    sp_cuda_band_dequantize(d_skel_bytes, cc->d_skel_scratch, bc, n_pos, s);

    // 2. Zero per-vec coeff slabs + scatter skeletons in.
    cudaMemsetAsync(cc->d_coeff_scratch, 0,
                    (size_t)pd * n_pos * sizeof(float), s);
    kernel_scatter_batch<<<n_pos, 128, 0, s>>>(
        cc->d_skel_scratch, sk_k,
        cc->d_skeleton_idx,
        cc->d_coeff_scratch, pd,
        sk_k, n_pos);

    if (n_res > 0) {
        // 3. Möbius predict — kernel_mobius_predict takes n_vecs, batches.
        sp_cuda_mobius_predict(cc->d_skel_scratch,
                                (float *)cc->d_pred_scratch,
                                cc->d_csr_offsets, cc->d_csr_skel_slot,
                                cc->d_csr_mu_sign,
                                sk_k, n_res, n_pos, s);

        // 4. Gather mag values from each slot into d_mag_scratch[n_pos].
        {
            int blk = 128;
            int grid = (n_pos + blk - 1) / blk;
            kernel_gather_mag_batch<<<grid, blk, 0, s>>>(
                src_base, bytes_per_pos, bc->total_bytes,
                cc->d_mag_scratch, n_pos);
        }

        // 5. Unpack residual levels for all n_pos in one batched dispatch.
        int res_bytes = (n_res * cc->residual_bits + 7) / 8;
        kernel_unpack_levels_batch<<<n_pos, 64, 0, s>>>(
            src_base, bytes_per_pos, bc->total_bytes + 4,
            n_res, cc->residual_bits,
            (uint8_t *)cc->d_levels_scratch, res_bytes, n_pos);

        // 6. Dequantize levels → deviation, batched per-vec mag.
        kernel_dequantize_residual_batch<<<n_pos, 128, 0, s>>>(
            cc->d_levels_scratch, n_res, cc->residual_bits,
            cc->d_mag_scratch, (float *)cc->d_dev_scratch, n_pos);

        // 7. Reconstruct residual positions across all n_pos vectors.
        //   No-spinor: per-vec scatter of pred + dev into coeff.
        //   Spinor   : strided memcpy2D of sheet bytes (sheet_bytes per vec
        //              at stride bytes_per_pos) into d_sheet_scratch, then
        //              kernel_spinor_reconstruct batched over n_pos.
        if (cc->use_spinor) {
            int sheet_bytes = (n_res + 7) / 8;
            cudaMemcpy2DAsync(cc->d_sheet_scratch, (size_t)sheet_bytes,
                              src_base + bc->total_bytes + 4 + res_bytes,
                              (size_t)bytes_per_pos,
                              (size_t)sheet_bytes, (size_t)n_pos,
                              cudaMemcpyDeviceToDevice, s);
            int block = (n_res < 256) ? n_res : 256;
            kernel_spinor_reconstruct<<<n_pos, block, 0, s>>>(
                cc->d_coeff_scratch,
                /*skel_vals*/ NULL,
                (const float *)cc->d_dev_scratch,
                cc->d_pred_scratch,
                cc->d_sheet_scratch,
                cc->d_residual_idx,
                pd, sk_k, n_res, n_pos, /*use_spinor*/ 1);
        } else {
            kernel_reconstruct_residual_batch<<<n_pos, 128, 0, s>>>(
                cc->d_pred_scratch, (const float *)cc->d_dev_scratch,
                cc->d_residual_idx, cc->d_coeff_scratch,
                n_res, pd, n_pos);
        }
    }

    // 8. Vilenkin inverse on all n_pos vecs.
    sp_cuda_vilenkin_inplace(cc->d_coeff_scratch, pd, n_pos,
                              cc->d_vilenkin_factors, cc->n_factors, s);

    // 9. Sqfree unpad each vec — kernel_sqfree_unpad takes n_vecs.
    kernel_sqfree_unpad<<<n_pos, 256, 0, s>>>(
        cc->d_coeff_scratch, d_out, hd, pd, n_pos);
}

void sp_cuda_sqfree_read_k_batch(const sp_cuda_sqfree_cache_t *cc,
                                  int layer, int head,
                                  int start_pos, int n_pos,
                                  float *d_k_out) {
    sp_cuda_sqfree_read_batch_one(cc, layer, head, start_pos, n_pos,
                                    d_k_out, &cc->k_bands,
                                    (const uint8_t *)cc->d_k_cache,
                                    cc->bytes_per_pos_k);
}

void sp_cuda_sqfree_read_v_batch(const sp_cuda_sqfree_cache_t *cc,
                                  int layer, int head,
                                  int start_pos, int n_pos,
                                  float *d_v_out) {
    sp_cuda_sqfree_read_batch_one(cc, layer, head, start_pos, n_pos,
                                    d_v_out, &cc->v_bands,
                                    (const uint8_t *)cc->d_v_cache,
                                    cc->bytes_per_pos_v);
}

// Public single-vector API. Ship-path style: one (layer, head, pos) at a time.
void sp_cuda_sqfree_write_k(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_k_vec) {
    sp_cuda_sqfree_write_one(cc, layer, head, pos, d_k_vec,
                              &cc->k_bands, (uint8_t *)cc->d_k_cache,
                              cc->bytes_per_pos_k);
}

void sp_cuda_sqfree_write_v(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_v_vec) {
    sp_cuda_sqfree_write_one(cc, layer, head, pos, d_v_vec,
                              &cc->v_bands, (uint8_t *)cc->d_v_cache,
                              cc->bytes_per_pos_v);
}

void sp_cuda_sqfree_read_k(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_k_out) {
    sp_cuda_sqfree_read_one(cc, layer, head, pos, d_k_out,
                             &cc->k_bands, (uint8_t *)cc->d_k_cache,
                             cc->bytes_per_pos_k);
}

void sp_cuda_sqfree_read_v(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_v_out) {
    sp_cuda_sqfree_read_one(cc, layer, head, pos, d_v_out,
                             &cc->v_bands, (uint8_t *)cc->d_v_cache,
                             cc->bytes_per_pos_v);
}

}  // extern "C"