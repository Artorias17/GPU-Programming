#include <stdio.h>

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    const unsigned int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    const unsigned int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x >= n)
        return;
    if (y >= m)
        return;

    const unsigned int i = n * y + x;
    C[i] = 0;

    unsigned int a = k * y, b = x;
    while (k > 0)
    {
        C[i] += A[a] * B[b];
        a++, b += n, k--;
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

    const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread blocks

    // INSERT CODE HERE

    dim3 dim_grid, dim_block;

    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim_grid = dim3(
        (n + dim_block.x - 1) / dim_block.x,
        (m + dim_block.y - 1) / dim_block.y
    );

    // Invoke CUDA kernel -----------------------------------------------------

    // INSERT CODE HERE

    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);
}
