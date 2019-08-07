/**
 * This kernel function converts an RBG image to grayscale.
 */

__kernel void rgb2gray(__constant unsigned char* inputRchannel,
                    __constant unsigned char* inputGchannel,
                    __constant unsigned char* inputBchannel,
                    __global unsigned char* outputImg){

    /**
     * Get work-item identifiers.
     */
    
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imgWidth = get_global_size(0);
    int index = (rowIndex * imgWidth) + colIndex;

    /**
     * Compute output pixel.
     * */

    outputImg[index] = (inputRchannel[index] + inputGchannel[index] + inputBchannel[index]) / 3;
}

/**
 * This kernel function convolves an image input_image[imgWidth, imgHeight]
 * with a mask of size maskSize.
 */

__kernel void filterImage(const unsigned int maskSize,
                            __global unsigned char* inputImg,
                            __constant float* mask,
                            __global unsigned char* outputImg){

    /**
     * Get work-item identifiers.
     */
    
    int colIndex = get_global_id(0);
    int rowIndex = get_global_id(1);
    int imgWidth = get_global_size(0);
    int imgHeight = get_global_size(1);
    int index = (rowIndex * imgWidth) + colIndex;

    /**
     * Check if the mask cannot be applied to the
     * current pixel.
     * */
    
    if(colIndex < maskSize/2 
    || rowIndex < maskSize/2
    || colIndex >= imgWidth - maskSize/2
    || rowIndex >= imgHeight - maskSize/2){
        outputImg[index] = 0;
        return;
    }

    /**
     * Apply mask based on the neighborhood of pixel inputImg(index).
     * */
    
    int outSum = 0;
    for(size_t k = 0; k < maskSize; k++){
        for(size_t l = 0; l < maskSize; l++){
            
            /**
             * Calculate the current mask index.
             */

            size_t maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize;

            /**
             * Compute output pixel.
             */

            size_t colIdx = colIndex - maskSize/2 + k;
            size_t rowIdx = rowIndex - maskSize/2 + l;
            outSum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
        }
    }

    /**
     * Write output pixel.
     * */

    if(outSum < 0){
        outputImg[index] = 0;
    } else if(outSum > 255){
        outputImg[index] = 255;
    }
    else{
        outputImg[index] = outSum;
    }
}

/**
 * This kernel function efficiently convolves an image input_image[imgWidth, imgHeight]
 * with a mask of size maskSize by caching submatrices from the input image 
 * in the device local memory.
 */

__kernel void filterImageWithCache(const unsigned int maskSize,
                            __global unsigned char* inputImg,
                            __constant float* mask,
                            __global unsigned char* outputImg){

    /**
     * Declare the size of each submatrix (it must be 
     * the same work-group size declared in the host code).
     */

    const int SUB_SIZE = 16;

    /**
     * Get work-item identifiers.
     */
    
    int colIndex = get_local_id(0);
    int rowIndex = get_local_id(1);
    int globalColIndex = get_global_id(0);
    int globalRowIndex = get_global_id(1);
    int imgWidth = get_global_size(0);
    int imgHeight = get_global_size(1);
    int index = (globalRowIndex * imgWidth) + globalColIndex;

    /**
     * Declare submatrix used to cache the input image on local memory.
     */

    __local unsigned char sub[SUB_SIZE][SUB_SIZE];

    /**
     * Synchronize all work-items in this work-group.
     */

    barrier(CLK_LOCAL_MEM_FENCE);

    /**
     * Load submatrix into local memory.
     */

    sub[rowIndex][colIndex] = inputImg[index];

    /**
     * Synchronize all work-items in this work-group.
     */

    barrier(CLK_LOCAL_MEM_FENCE);

    /**
     * Check if the mask cannot be applied to the
     * current pixel.
     * */
    
    if(globalColIndex < maskSize/2 
    || globalRowIndex < maskSize/2
    || globalColIndex >= imgWidth - maskSize/2
    || globalRowIndex >= imgHeight - maskSize/2){
        outputImg[index] = 0;
        return;
    }

    /**
     * Apply mask based on the neighborhood of the pixel inputImg(index).
     * */
    
    int outSum = 0;
    for(size_t k = 0; k < maskSize; k++){
        for(size_t l = 0; l < maskSize; l++){
            
            /**
             * Calculate the current mask index.
             */

            size_t maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize;

            /**
             * Check if the current input pixel is in the local memory
             * and compute output.
             */

            size_t colIdx = colIndex - maskSize/2 + k;
            size_t rowIdx = rowIndex - maskSize/2 + l;
            if(colIdx >= 0 && colIdx < maskSize && rowIdx >= 0 && rowIdx < maskSize){
                outSum += sub[rowIdx][colIdx] * mask[maskIdx];
            }

            /**
             * Read the current input pixel from the global memory
             * and compute output.
             */

            else{
                size_t colIdx = globalColIndex - maskSize/2 + k;
                size_t rowIdx = globalRowIndex - maskSize/2 + l;
                outSum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
            }
            
        }
    }

    /**
     * Write output pixel.
     * */

    if(outSum < 0){
        outputImg[index] = 0;
    } else if(outSum > 255){
        outputImg[index] = 255;
    } else{
        outputImg[index] = outSum;
    }
}