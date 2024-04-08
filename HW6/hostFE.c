#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hostFE.h"
#include "helper.h"

#define GROUP_SIZE_H 16
#define GROUP_SIZE_W 32


void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int halfFilterWidth = filterWidth / 2;
    int filterSize = filterWidth * filterWidth;
    int imageSize  = imageHeight * imageWidth;

    int sharedHeight = GROUP_SIZE_H + 2 * halfFilterWidth;
    int sharedWidth  = GROUP_SIZE_W + 2 * halfFilterWidth;
    int sharedSize   = sharedHeight * sharedWidth;

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, NULL);

    cl_mem devFilterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), filter, NULL);
    cl_mem devInputBuffer  = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imageSize  * sizeof(float), inputImage, NULL);
    cl_mem devOutputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY  , imageSize  * sizeof(float), NULL, NULL);
    // clEnqueueWriteBuffer(queue, devFilterBuffer, CL_FALSE, 0, filterSize * sizeof(float), filter    , 0, NULL, NULL);
    // clEnqueueWriteBuffer(queue, devInputBuffer , CL_TRUE , 0, imageSize  * sizeof(float), inputImage, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem)            , (void*)&devFilterBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem)            , (void*)&devInputBuffer );
    clSetKernelArg(kernel, 2, sizeof(cl_mem)            , (void*)&devOutputBuffer);
    clSetKernelArg(kernel, 3, sizeof(int)               , (void*)&filterWidth    );
    clSetKernelArg(kernel, 4, sizeof(int)               , (void*)&imageHeight    );
    clSetKernelArg(kernel, 5, sizeof(int)               , (void*)&imageWidth     );
    clSetKernelArg(kernel, 6, sizeof(int)               , (void*)&sharedWidth    );
    clSetKernelArg(kernel, 7, sizeof(float) * sharedSize, NULL                   );

    size_t localws[2]  = {
        GROUP_SIZE_W,
        GROUP_SIZE_H 
    };
    size_t globalws[2] = {
        imageWidth  + (imageWidth  % GROUP_SIZE_W == 0? 0 : GROUP_SIZE_W - imageWidth  % GROUP_SIZE_W),
        imageHeight + (imageHeight % GROUP_SIZE_H == 0? 0 : GROUP_SIZE_H - imageHeight % GROUP_SIZE_H)
    };
    clEnqueueNDRangeKernel(
        queue, kernel, 2, NULL, globalws, localws, 0, NULL, NULL
    );

    clEnqueueReadBuffer(queue, devOutputBuffer, CL_TRUE, 0, imageSize * sizeof(float), outputImage, 0, NULL, NULL);

    clFinish(queue);

    // clReleaseMemObject(devFilterBuffer);
    // clReleaseMemObject(devInputBuffer);
    // clReleaseMemObject(devOutputBuffer);
    // clReleaseKernel(kernel);
}