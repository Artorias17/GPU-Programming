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

  const unsigned int global_x = (blockDim.x * blockIdx.x) + threadIdx.x;
  const unsigned int global_y = (blockDim.y * blockIdx.y) + threadIdx.y;

  const unsigned int FILTER_RADIUS = FILTER_SIZE / 2;

  __shared__ float N_tile[BLOCK_SIZE][BLOCK_SIZE];

  // Load tile to shared memory with zero padding
  N_tile[threadIdx.y][threadIdx.x] = 0;
  if (global_x < N.width && global_y < N.height)
  {
    N_tile[threadIdx.y][threadIdx.x] = N.elements[global_y * N.width + global_x];
  }

  __syncthreads();

  // Apply convolution only for valid output region

  if (global_x < P.width && global_y < P.height)
  {
    float sum = 0;
    for (int fy = 0; fy < FILTER_SIZE; fy++)
    {
      for (int fx = 0; fx < FILTER_SIZE; fx++)
      {
        // Get tile position for filter relative to the threadIdx 
        int tile_x = threadIdx.x + fx - FILTER_RADIUS;
        int tile_y = threadIdx.y + fy - FILTER_RADIUS;

        if (tile_x >= 0 && tile_x < BLOCK_SIZE && tile_y >= 0 && tile_y < BLOCK_SIZE)
        {
          sum += M_c[fy][fx] * N_tile[tile_y][tile_x];
        }
      }
    }

    P.elements[global_y * P.width + global_x] = sum;
  }
}
