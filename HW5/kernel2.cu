#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


__device__ inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i)
  {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, size_t pitch_size, int maxIterations, int* output_dev) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= resX || thisY >= resY)
        return;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    *((int*)((char*)output_dev + thisY * pitch_size) + thisX) = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t mem_size, pitch_size;

    int* output_dev;
    cudaMallocPitch(&output_dev, &pitch_size, resX * sizeof(int), resY);

    mem_size = pitch_size * resY;

    int* output_host;
    cudaMallocHost(&output_host, mem_size, cudaHostAllocDefault);

    dim3 grid(
        resX / BLOCK_SIZE_X + (resX % BLOCK_SIZE_X == 0? 0 : 1),
        resY / BLOCK_SIZE_Y + (resY % BLOCK_SIZE_Y == 0? 0 : 1)
    );
    dim3 block(
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y
    );
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, resX, resY, pitch_size, maxIterations, output_dev);

    cudaMemcpy(output_host, output_dev, mem_size, cudaMemcpyDeviceToHost);
    for (int j = 0; j < resY; j++){
        memcpy(img + j * resX, (char*)output_host + j * pitch_size, resX * sizeof(int));
    }

    cudaFree(output_dev);
    cudaFreeHost(output_host);
}
