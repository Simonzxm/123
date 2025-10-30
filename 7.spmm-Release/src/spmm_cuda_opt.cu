#include <cuda_runtime.h>
#include <iostream>
#include <spmm_cuda_opt.h>


// Each warp computes one output row across N columns in tiles of 32 columns (1 per lane)
__global__ void spmm_kernel_opt_device(const int *__restrict__ ptr, const int *__restrict__ idx, const float *__restrict__ val,
									   const float *__restrict__ vin, float *__restrict__ vout, int m, int n, int k)
{
	const int lane = threadIdx.x & 31;                // lane id within warp
	const int warpInBlock = threadIdx.x >> 5;         // warp id within block
	const int warpsPerBlock = blockDim.x >> 5;
	// global warp id for grid-stride over rows
	int warpGlobal = blockIdx.x * warpsPerBlock + warpInBlock;
	unsigned full_mask = 0xffffffffu;

	// Grid-stride loop over rows (by warps)
	for (int row = warpGlobal; row < m; row += gridDim.x * warpsPerBlock)
	{
		const int row_start = ptr[row];
		const int row_end   = ptr[row + 1];

		// Tile over dense dimension N in chunks of 32 columns per warp iteration
		for (int jt = 0; jt < n; jt += 32)
		{
			const int cj = jt + lane;           // column computed by this lane

			float acc = 0.0f;

			// Iterate over nonzeros in this row
			for (int p = row_start; p < row_end; ++p)
			{
				int kcol = 0;
				float aval = 0.0f;
				if (lane == 0)
				{
					kcol = idx[p];
					aval = val[p];
				}
				// Broadcast the nonzero to the whole warp
				kcol = __shfl_sync(full_mask, kcol, 0);
				aval = __shfl_sync(full_mask, aval, 0);

				// Accumulate one column per lane
				if (cj < n && kcol >= 0 && kcol < k)
				{
					acc = fmaf(aval, vin[kcol * n + cj], acc);
				}
			}

			// Write results
			if (cj < n)
			{
				vout[row * n + cj] = acc;
			}
		}
	}
}


void spmm_cuda_opt(int *d_ptr, int *d_idx, float *d_val, float *d_vin, float *d_vout, int m, int n,int k)
{
	// Kernel configuration: use multiple warps per block; one warp processes one sparse row
	const int threadsPerBlock = 256; // 8 warps per block
	const int warpsPerBlock = threadsPerBlock / 32;
	dim3 block(threadsPerBlock);
	dim3 grid((m + warpsPerBlock - 1) / warpsPerBlock);

	spmm_kernel_opt_device<<<grid, block>>>(d_ptr, d_idx, d_val, d_vin, d_vout, m, n, k);
}

