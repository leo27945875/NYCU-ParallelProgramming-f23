__kernel void convolution(
    __constant float *filter,
    __global   float *inputImage,
    __global   float *outputImage,
    int filterWidth, 
    int imageHeight, 
    int imageWidth,
    int sharedWidth,
    __local float *shared_buf
){
    int i = get_global_id(1);
    int j = get_global_id(0);
    if (i >= imageHeight || j >= imageWidth) 
        return;

    int halffilterSize = filterWidth / 2;
    int wLocalSize = get_local_size(0);
    int hLocalSize = get_local_size(1);
    int wLocalID   = get_local_id(0);
    int hLocalID   = get_local_id(1);

    int wLimitRight  = (get_group_id(0) + 1) * wLocalSize + halffilterSize;
    int hLimitBottom = (get_group_id(1) + 1) * hLocalSize + halffilterSize;

    for (int h = i - halffilterSize, a = hLocalID; h < hLimitBottom; h+=hLocalSize, a+=hLocalSize){
    for (int w = j - halffilterSize, b = wLocalID; w < wLimitRight ; w+=wLocalSize, b+=wLocalSize){
        if (h < imageHeight && h >= 0 && w < imageWidth && w >= 0) 
            shared_buf[a * sharedWidth + b] = inputImage[h * imageWidth + w];
        else
            shared_buf[a * sharedWidth + b] = 0;
    }}

    barrier(CLK_LOCAL_MEM_FENCE);

    int locali = hLocalID + halffilterSize;
    int localj = wLocalID + halffilterSize;

    float sum = 0.; 
    int k, l;
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            sum += shared_buf[(locali + k) * sharedWidth + localj + l] *
                    filter[(k + halffilterSize) * filterWidth +
                            l + halffilterSize];
        }
    }
    outputImage[i * imageWidth + j] = sum;
}
