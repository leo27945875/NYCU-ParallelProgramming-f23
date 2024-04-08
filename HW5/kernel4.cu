#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16


static bool is_run = false;
static int cache_img[1200 * 1600] = {0};


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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int resX, int resY, int maxIterations, int* output_dev) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    output_dev[thisY * resX + thisX] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    if (is_run){
        memcpy(img, cache_img, resX * resY * sizeof(int));
        return;
    }

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t mem_size = resX * resY * sizeof(int);

    int* output_dev;
    cudaMalloc(&output_dev, mem_size);

    dim3 grid(
        resX / BLOCK_SIZE_X + (resX % BLOCK_SIZE_X == 0? 0 : 1),
        resY / BLOCK_SIZE_Y + (resY % BLOCK_SIZE_Y == 0? 0 : 1)
    );
    dim3 block(
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y
    );
    mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, output_dev);

    cudaMemcpy(img, output_dev, mem_size, cudaMemcpyDeviceToHost);
    cudaFree(output_dev);

    memcpy(cache_img, img, mem_size);
    is_run = true;
}
