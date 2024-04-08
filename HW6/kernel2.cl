__kernel void convolution(
    __global float *filter,
    __global float *inputImage,
    __global float *outputImage,
    int filterWidth, 
    int imageHeight, 
    int imageWidth
){
    int i = get_global_id(1);
    int j = get_global_id(0);

    if (i >= imageHeight || j >= imageWidth) 
        return;

    int k, l;
    int sum = 0; 
    int halffilterSize = filterWidth / 2;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (i + k >= 0 && i + k < imageHeight &&
                j + l >= 0 && j + l < imageWidth)
            {
                sum += inputImage[(i + k) * imageWidth + j + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    outputImage[i * imageWidth + j] = sum;
}
