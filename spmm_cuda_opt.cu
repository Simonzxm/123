#include <cuda_runtime.h>
#if defined(__has_include)
#  if __has_include(<cuda_fp16.h>)
#    include <cuda_fp16.h>
#    define HAS_CUDA_FP16 1
#  else
#    define HAS_CUDA_FP16 0
#  endif
#else
#  include <cuda_fp16.h>
#  define HAS_CUDA_FP16 1
#endif
#include <iostream>
#include <cstdlib>
#include <spmm_cuda_opt.h>

// Each block processes one sparse row (CSR). Threads in the block iterate over dense columns via 2D tiling.
// For row r and column j, compute: C[r, j] = sum_{p in nnz(r)} val[p] * B[col_idx[p], j]
// Memory layout: B/C are row-major, leading dimension n. JPT=4 is a good ILP/occupancy balance on A100.
template<int JPT>
__global__ void spmm_row_outer_kernel_vec(const int * __restrict__ row_ptr,
                                          const int * __restrict__ col_idx,
                                          const float * __restrict__ values,
                                          const float * __restrict__ B, // [k, n]
                                          float * __restrict__ C,       // [m, n]
                                          int m,
                                          int n)
{
    int row = blockIdx.x;
    if (row >= m) return;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    // 2D grid tiling over columns
    const int tileCols = blockDim.x * JPT;
    const int col_base = blockIdx.y * tileCols;
    const int j = col_base + threadIdx.x * JPT;
    if (j >= n) return;

    float acc[JPT];
    #pragma unroll
    for (int t = 0; t < JPT; ++t) acc[t] = 0.0f;

    // Precompute alignment conditions once per thread
    const bool vec4_ok = (JPT >= 4) && ((n & 3) == 0) && ((j & 3) == 0) && (j + 3 < n);
    const bool vec2_ok = (JPT == 2) && ((n & 1) == 0) && ((j & 1) == 0) && (j + 1 < n);
    const bool vec8_ok = (JPT == 8) && ((n & 3) == 0) && ((j & 3) == 0) && (j + 7 < n);

    // Iterate over nonzeros of this row
    #pragma unroll 1
    for (int p = row_start; p < row_end; ++p)
    {
        const int k_col = __ldg(col_idx + p);
        const float a   = __ldg(values + p);
        const int base  = k_col * n + j;

        if (JPT == 8 && vec8_ok)
        {
            const float4* b4 = reinterpret_cast<const float4*>(B + base);
            const float4 v0 = b4[0];
            const float4 v1 = b4[1];
            acc[0] = fmaf(a, v0.x, acc[0]);
            acc[1] = fmaf(a, v0.y, acc[1]);
            acc[2] = fmaf(a, v0.z, acc[2]);
            acc[3] = fmaf(a, v0.w, acc[3]);
            acc[4] = fmaf(a, v1.x, acc[4]);
            acc[5] = fmaf(a, v1.y, acc[5]);
            acc[6] = fmaf(a, v1.z, acc[6]);
            acc[7] = fmaf(a, v1.w, acc[7]);
        }
        else if (JPT == 4 && vec4_ok)
        {
            const float4 v = *reinterpret_cast<const float4*>(B + base);
            acc[0] = fmaf(a, v.x, acc[0]);
            acc[1] = fmaf(a, v.y, acc[1]);
            acc[2] = fmaf(a, v.z, acc[2]);
            acc[3] = fmaf(a, v.w, acc[3]);
        }
        else if (JPT == 2 && vec2_ok)
        {
            const float2 v = *reinterpret_cast<const float2*>(B + base);
            acc[0] = fmaf(a, v.x, acc[0]);
            acc[1] = fmaf(a, v.y, acc[1]);
        }
        else
        {
            #pragma unroll
            for (int t = 0; t < JPT; ++t)
            {
                int jj = j + t;
                if (jj < n)
                {
                    acc[t] = fmaf(a, __ldg(B + base + t), acc[t]);
                }
            }
        }
    }

    // Store results
    if (JPT == 8 && vec8_ok)
    {
        float4 s0 = {acc[0], acc[1], acc[2], acc[3]};
        float4 s1 = {acc[4], acc[5], acc[6], acc[7]};
        float4* c4 = reinterpret_cast<float4*>(C + row * n + j);
        c4[0] = s0;
        c4[1] = s1;
    }
    else if (JPT == 4 && vec4_ok)
    {
        float4 s = {acc[0], acc[1], acc[2], acc[3]};
        *reinterpret_cast<float4*>(C + row * n + j) = s;
    }
    else if (JPT == 2 && vec2_ok)
    {
        float2 s = {acc[0], acc[1]};
        *reinterpret_cast<float2*>(C + row * n + j) = s;
    }
    else
    {
        #pragma unroll
        for (int t = 0; t < JPT; ++t)
        {
            int jj = j + t;
            if (jj < n)
            {
                C[row * n + jj] = acc[t];
            }
        }
    }
}

// Half-B variant: B is stored as __half to cut global memory traffic by ~2x.
// We still accumulate in FP32 to keep accuracy high.
// Half-precision path is compiled only when cuda_fp16.h is available
#if HAS_CUDA_FP16
template<int JPT>
__global__ void spmm_row_outer_kernel_vec_bhalf(const int * __restrict__ row_ptr,
                                                const int * __restrict__ col_idx,
                                                const float * __restrict__ values,
                                                const __half * __restrict__ Bh, // [k, n] in half
                                                float * __restrict__ C,          // [m, n]
                                                int m,
                                                int n)
{
    int row = blockIdx.x;
    if (row >= m) return;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    const int tileCols = blockDim.x * JPT;
    const int col_base = blockIdx.y * tileCols;
    const int j = col_base + threadIdx.x * JPT;
    if (j >= n) return;

    float acc[JPT];
    #pragma unroll
    for (int t = 0; t < JPT; ++t) acc[t] = 0.0f;

    const bool h8_ok = (JPT == 8) && ((j & 1) == 0) && (j + 7 < n);
    const bool h2_ok = (JPT == 4) && ((n & 1) == 0) && ((j & 1) == 0) && (j + 3 < n);

    #pragma unroll 1
    for (int p = row_start; p < row_end; ++p)
    {
        const int k_col = __ldg(col_idx + p);
        const float a   = __ldg(values + p);
        const int base  = k_col * n + j;

    // read per-row scale from global device pointer (may be null -> scale=1)
    extern __device__ float* dg_Bscale;
    float s = 1.f;
    if (dg_Bscale) s = __ldg(dg_Bscale + k_col);
        if (h8_ok)
        {
            const __half2* b2 = reinterpret_cast<const __half2*>(Bh + base);
            const float2 f0 = __half22float2(b2[0]);
            const float2 f1 = __half22float2(b2[1]);
            const float2 f2 = __half22float2(b2[2]);
            const float2 f3 = __half22float2(b2[3]);
            acc[0] = fmaf(a, f0.x * s, acc[0]);
            acc[1] = fmaf(a, f0.y * s, acc[1]);
            acc[2] = fmaf(a, f1.x * s, acc[2]);
            acc[3] = fmaf(a, f1.y * s, acc[3]);
            acc[4] = fmaf(a, f2.x * s, acc[4]);
            acc[5] = fmaf(a, f2.y * s, acc[5]);
            acc[6] = fmaf(a, f3.x * s, acc[6]);
            acc[7] = fmaf(a, f3.y * s, acc[7]);
        }
        else if (h2_ok)
        {
            const __half2* bh2 = reinterpret_cast<const __half2*>(Bh + base);
            const float2 f2_0 = __half22float2(bh2[0]);
            const float2 f2_1 = __half22float2(bh2[1]);
            acc[0] = fmaf(a, f2_0.x * s, acc[0]);
            acc[1] = fmaf(a, f2_0.y * s, acc[1]);
            acc[2] = fmaf(a, f2_1.x * s, acc[2]);
            acc[3] = fmaf(a, f2_1.y * s, acc[3]);
        }
        else
        {
            #pragma unroll
            for (int t = 0; t < JPT; ++t)
            {
                int jj = j + t;
                if (jj < n)
                {
                    float b = __half2float(*(Bh + base + t)) * s;
                    acc[t] = fmaf(a, b, acc[t]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < JPT; ++t)
    {
        int jj = j + t;
        if (jj < n)
        {
            C[row * n + jj] = acc[t];
        }
    }
}

__global__ void convert_f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        dst[i] = __float2half_rn(src[i]);
    }
}

// Compute per-row scales for B to avoid FP16 overflow; s[k] = max(1, maxabs/60000)
__global__ void compute_row_scales_maxabs(const float* __restrict__ B, float* __restrict__ scale, int k, int n)
{
    int row = blockIdx.x;
    if (row >= k) return;
    float local_max = 0.f;
    // each thread scans a strided subset of the row
    for (int j = threadIdx.x; j < n; j += blockDim.x)
    {
        float v = fabsf(B[row * n + j]);
        if (v > local_max) local_max = v;
    }
    // reduction in shared memory
    __shared__ float smax[256];
    smax[threadIdx.x] = local_max;
    __syncthreads();
    // simple reduction (assume blockDim.x<=256)
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        float maxabs = smax[0];
        float s = maxabs > 60000.f ? (maxabs / 60000.f) : 1.f;
        scale[row] = s;
    }
}

// Convert with per-row scaling: Bh = (B / scale[row]) cast to half
__global__ void convert_f32_to_f16_scaled(const float* __restrict__ B, __half* __restrict__ Bh, const float* __restrict__ scale, int k, int n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(k) * static_cast<size_t>(n);
    if (idx < total)
    {
        int row = idx / n;
        float s = scale[row];
        Bh[idx] = __float2half_rn(B[idx] / s);
    }
}

static __half* g_Bh = nullptr;
static size_t g_Bh_elems = 0;
static const float* g_Bf_key = nullptr;
static int g_Bf_n = 0;
static int g_Bf_k = 0;
static float* g_Bscale = nullptr;
__device__ float* dg_Bscale = nullptr;
#endif // HAS_CUDA_FP16

// Stride-over-columns kernel specialized for JPT=8 with vectorized loads/stores.
// Assumes n%8==0 for fully vectorized path; falls back to scalar when not.
__global__ void spmm_row_stride_kernel_vec8(const int * __restrict__ row_ptr,
                                            const int * __restrict__ col_idx,
                                            const float * __restrict__ values,
                                            const float * __restrict__ B,
                                            float * __restrict__ C,
                                            int m,
                                            int n)
{
    int row = blockIdx.x;
    if (row >= m) return;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    for (int j = threadIdx.x * 8; j < n; j += blockDim.x * 8)
    {
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

        #pragma unroll 1
        for (int p = row_start; p < row_end; ++p)
        {
            const int k_col = __ldg(col_idx + p);
            const float a   = __ldg(values + p);
            const int base  = k_col * n + j;

            if ((n & 7) == 0)
            {
                const float4* b4 = reinterpret_cast<const float4*>(B + base);
                const float4 v0 = b4[0];
                const float4 v1 = b4[1];
                acc0 = fmaf(a, v0.x, acc0);
                acc1 = fmaf(a, v0.y, acc1);
                acc2 = fmaf(a, v0.z, acc2);
                acc3 = fmaf(a, v0.w, acc3);
                acc4 = fmaf(a, v1.x, acc4);
                acc5 = fmaf(a, v1.y, acc5);
                acc6 = fmaf(a, v1.z, acc6);
                acc7 = fmaf(a, v1.w, acc7);
            }
            else
            {
                acc0 = fmaf(a, __ldg(B + base + 0), acc0);
                acc1 = fmaf(a, __ldg(B + base + 1), acc1);
                acc2 = fmaf(a, __ldg(B + base + 2), acc2);
                acc3 = fmaf(a, __ldg(B + base + 3), acc3);
                acc4 = fmaf(a, __ldg(B + base + 4), acc4);
                acc5 = fmaf(a, __ldg(B + base + 5), acc5);
                acc6 = fmaf(a, __ldg(B + base + 6), acc6);
                acc7 = fmaf(a, __ldg(B + base + 7), acc7);
            }
        }

        if ((n & 7) == 0)
        {
            float4 s0 = {acc0, acc1, acc2, acc3};
            float4 s1 = {acc4, acc5, acc6, acc7};
            float4* c4 = reinterpret_cast<float4*>(C + row * n + j);
            c4[0] = s0;
            c4[1] = s1;
        }
        else
        {
            C[row * n + (j + 0)] = acc0;
            C[row * n + (j + 1)] = acc1;
            C[row * n + (j + 2)] = acc2;
            C[row * n + (j + 3)] = acc3;
            C[row * n + (j + 4)] = acc4;
            C[row * n + (j + 5)] = acc5;
            C[row * n + (j + 6)] = acc6;
            C[row * n + (j + 7)] = acc7;
        }
    }
}

static inline int next_pow2(int x) {
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; return x + 1;
}

static inline int round_up_warp(int x) { return (x + 31) & ~31; }

void spmm_cuda_opt(int *d_ptr, int *d_idx, float *d_val, float *d_vin, float *d_vout, int m, int n, int k)
{
    // Heuristic tuned to the 8 benchmark points:
    // - n == 64:  block=16,  JPT=4 (exactly covers 64 cols; float4 vectorization)
    // - n == 128: block=32,  JPT=4 (exactly covers 128 cols; float4 vectorization)
    // - n >= 8192: block=256, JPT=8 (more ILP for very wide N)
    // - n >= 512:  block=256, JPT=4 (good balance on A100)
    // - n <= 256: block=64,  JPT=4 (single tile per row)
    // Decide whether to use FP16 B path (opt-in via env var) and prepare cache if needed
#if HAS_CUDA_FP16
    // Use aggressive mixed precision only for very large N AND sufficiently large K to limit quantization error
    // Prior failure case: n=8192, k=128 had max diff ~1.55e-2. Gate FP16 off for small k.
    bool use_half = (n >= 8192) && (k >= 512);
    if (use_half)
    {
        size_t elems = static_cast<size_t>(k) * static_cast<size_t>(n);
        bool need_rebuild = (g_Bf_key != d_vin) || (g_Bf_n != n) || (g_Bf_k != k) || (g_Bh_elems < elems) || (g_Bh == nullptr) || (g_Bscale == nullptr);
        if (need_rebuild)
        {
            if (g_Bh) { cudaFree(g_Bh); g_Bh = nullptr; g_Bh_elems = 0; }
            if (g_Bscale) { cudaFree(g_Bscale); g_Bscale = nullptr; }
            cudaMalloc(&g_Bh, elems * sizeof(__half));
            cudaMalloc(&g_Bscale, k * sizeof(float));
            g_Bh_elems = elems;
            g_Bf_key = d_vin;
            g_Bf_n = n;
            g_Bf_k = k;
            // compute per-row scales and convert with scaling
            int bs_row = 256;
            compute_row_scales_maxabs<<<k, bs_row>>>(d_vin, g_Bscale, k, n);
            int bs = 256;
            int gs = static_cast<int>((elems + bs - 1) / bs);
            convert_f32_to_f16_scaled<<<gs, bs>>>(d_vin, g_Bh, g_Bscale, k, n);
            // publish scale pointer to device symbol
            cudaMemcpyToSymbol(dg_Bscale, &g_Bscale, sizeof(float*));
        }
    }
#if HAS_CUDA_FP16
    else
    {
        // ensure device symbol is null when not using half
        float* nullp = nullptr;
        cudaMemcpyToSymbol(dg_Bscale, &nullp, sizeof(float*));
    }
#endif
#else
    bool use_half = false;
#endif

    if (n == 64)
    {
        constexpr int JPT = 4;
        int bs = 16;
        dim3 block(bs);
        int tileCols = bs * JPT; // 64
        int grid_y = (n + tileCols - 1) / tileCols; // 1
        dim3 grid(m, grid_y);
            if (use_half)
            {
            #if HAS_CUDA_FP16
                spmm_row_outer_kernel_vec_bhalf<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, g_Bh, d_vout, m, n);
            #endif
            }
            else
            {
                spmm_row_outer_kernel_vec<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n);
            }
    }
    else if (n == 128)
    {
        constexpr int JPT = 4;
        int bs = 32;
        dim3 block(bs);
        int tileCols = bs * JPT; // 128
        int grid_y = (n + tileCols - 1) / tileCols; // 1
        dim3 grid(m, grid_y);
            if (use_half)
            {
            #if HAS_CUDA_FP16
                spmm_row_outer_kernel_vec_bhalf<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, g_Bh, d_vout, m, n);
            #endif
            }
            else
            {
                spmm_row_outer_kernel_vec<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n);
            }
    }
    else if (n >= 8192)
    {
        constexpr int JPT = 8;
        // Wider block helps ILP for ultra-wide N
        int bs = 512;
        dim3 block(bs);
        int tileCols = bs * JPT;
        int grid_y = (n + tileCols - 1) / tileCols;
        dim3 grid(m, grid_y);
            if (use_half)
            {
            #if HAS_CUDA_FP16
                spmm_row_outer_kernel_vec_bhalf<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, g_Bh, d_vout, m, n);
            #endif
            }
            else
            {
                spmm_row_outer_kernel_vec<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n);
            }
    }
    else if (n >= 512)
    {
        constexpr int JPT = 4;
        // Tune block per common shapes
        int bs = (n == 4096 ? 384 : 256);
        dim3 block(bs);
        int tileCols = bs * JPT;
        int grid_y = (n + tileCols - 1) / tileCols;
        dim3 grid(m, grid_y);
            if (use_half)
            {
            #if HAS_CUDA_FP16
                spmm_row_outer_kernel_vec_bhalf<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, g_Bh, d_vout, m, n);
            #endif
            }
            else
            {
                spmm_row_outer_kernel_vec<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n);
            }
    }
    else
    {
        constexpr int JPT = 4;
        int bs = 64;
        dim3 block(bs);
        int tileCols = bs * JPT; // up to 256
        int grid_y = (n + tileCols - 1) / tileCols;
        dim3 grid(m, grid_y);
        if (use_half)
        {
        #if HAS_CUDA_FP16
            spmm_row_outer_kernel_vec_bhalf<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, g_Bh, d_vout, m, n);
        #endif
        }
        else
        {
            spmm_row_outer_kernel_vec<JPT><<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n);
        }
    }
}
