#include "kernel.cu"
#include "support.h"
#include <stdio.h>

int main(int argc, char **argv) {

  Timer timer;
  cudaError_t cuda_ret;

  // Initialize host variables ----------------------------------------------

  printf("\nSetting up the problem...");
  fflush(stdout);
  startTime(&timer);

  unsigned int n;
  if (argc == 1) {
    n = 10000;
  } else if (argc == 2) {
    n = atoi(argv[1]);
  } else {
    printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
    exit(0);
  }

  float *A_h = (float *)malloc(sizeof(float) * n);
  for (unsigned int i = 0; i < n; i++) {
    A_h[i] = (rand() % 100) / 100.00;
  }

  float *B_h = (float *)malloc(sizeof(float) * n);
  for (unsigned int i = 0; i < n; i++) {
    B_h[i] = (rand() % 100) / 100.00;
  }

  float *C_h = (float *)malloc(sizeof(float) * n);

  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));
  printf("    Vector size = %u\n", n);

  // Allocate device variables ----------------------------------------------

  printf("Allocating device variables...");
  fflush(stdout);
  startTime(&timer);

  float *A_d, *B_d, *C_d;
  cudaMalloc(&A_d, sizeof(float) * n);
  cudaMalloc(&B_d, sizeof(float) * n);
  cudaMalloc(&C_d, sizeof(float) * n);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Copy host variables to device ------------------------------------------

  printf("Copying data from host to device...");
  fflush(stdout);
  startTime(&timer);

  cudaMemcpy(A_d, A_h, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeof(float) * n, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Launch kernel ----------------------------------------------------------

  printf("Launching kernel...");
  fflush(stdout);
  startTime(&timer);

  int numberOfThreadsPerBlock = 256;
  int numberOfBlocks = (n + numberOfThreadsPerBlock - 1) / numberOfThreadsPerBlock;
  vecAddKernel<<<numberOfBlocks, numberOfThreadsPerBlock>>>(A_d, B_d, C_d, n);

  cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess)
    FATAL("Unable to launch kernel");
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Copy device variables from host ----------------------------------------

  printf("Copying data from device to host...");
  fflush(stdout);
  startTime(&timer);

  cudaMemcpy(C_h, C_d, sizeof(float) * n, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  stopTime(&timer);
  printf("%f s\n", elapsedTime(timer));

  // Verify correctness -----------------------------------------------------

  printf("Verifying results...");
  fflush(stdout);

  verify(A_h, B_h, C_h, n);

  // Free memory ------------------------------------------------------------

  free(A_h);
  free(B_h);
  free(C_h);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return 0;
}
