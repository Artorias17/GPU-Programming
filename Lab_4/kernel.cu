#include "support.h"

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
  /********************************************************************
  Determine input and output indexes of each thread
  Load a tile of the input image to shared memory
  Apply the filter on the input image tile
  Write the compute values to the output image at the correct indexes
  ********************************************************************/

  // INSERT KERNEL CODE HERE

  __shared__ float N_tile[BLOCK_SIZE][BLOCK_SIZE];

  int blockId_x = blockIdx.x;
  int blockId_y = blockIdx.y;
  int threadId_x = threadIdx.x;
  int threadId_y = threadIdx.y;
  int blockDim_x = blockDim.x;
  int blockDim_y = blockDim.y;

  int FILTER_RADIUS = FILTER_SIZE / 2;

  // Input starting position for the tile (with padding)
  int in_y_start = blockId_y * blockDim_y - FILTER_RADIUS;
  int in_x_start = blockId_x * blockDim_x - FILTER_RADIUS;

  // Load padded tile into shared memory
  // Each thread loads multiple elements
  for (int dy = 0; dy <= 1; dy++)
  {
    for (int dx = 0; dx <= 1; dx++)
    {
      int shared_y = threadId_y + dy * TILE_SIZE;
      int shared_x = threadId_x + dx * TILE_SIZE;

      if (shared_y < BLOCK_SIZE && shared_x < BLOCK_SIZE)
      {
        int global_y = in_y_start + shared_y;
        int global_x = in_x_start + shared_x;

        N_tile[shared_y][shared_x] = 0;

        if (global_y >= 0 && global_y < N.height &&
            global_x >= 0 && global_x < N.width)
        {
          N_tile[shared_y][shared_x] = N.elements[global_y * N.width + global_x];
        }
      }
    }
  }

  __syncthreads();

  // Compute convolution for output pixels
  int out_y = blockId_y * TILE_SIZE + threadId_y;
  int out_x = blockId_x * TILE_SIZE + threadId_x;

  if (out_y < P.height && out_x < P.width)
  {
    float sum = 0;
    for (int fy = 0; fy < FILTER_SIZE; fy++)
    {
      for (int fx = 0; fx < FILTER_SIZE; fx++)
      {
        int shared_y = threadId_y + fy;
        int shared_x = threadId_x + fx;
        sum += M_c[fy][fx] * N_tile[shared_y][shared_x];
      }
    }
    P.elements[out_y * P.width + out_x] = sum;
  }
}
