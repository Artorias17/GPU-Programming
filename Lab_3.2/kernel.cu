#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    const unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    float sum = 0;
    const int tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tile_count; tile++) {
        A_tile[threadIdx.y][threadIdx.x] = 0;
        B_tile[threadIdx.y][threadIdx.x] = 0;

        const unsigned int A_i = k*y + TILE_SIZE*tile + threadIdx.x;
        const unsigned int B_i = n*(TILE_SIZE*tile + threadIdx.y)  + x;

        if (A_i < m * k) {
            A_tile[threadIdx.y][threadIdx.x] = A[A_i];
        }

        if (B_i < k * n) {
            B_tile[threadIdx.y][threadIdx.x] = B[B_i];
        }

        __syncthreads();

        // Partial dot product
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < n && y < m) {
        C[n * y + x] = sum;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    
    dim3 dim_grid, dim_block;

    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim_grid = dim3(
        (n + dim_block.x - 1) / dim_block.x,
        (m + dim_block.y - 1) / dim_block.y
    );

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);
}


