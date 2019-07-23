#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <time.h>

#include "CImg.h"
using namespace cimg_library;

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

void seqRgb2Gray(unsigned int imgWidth,
                        unsigned int imgHeight,
                        unsigned char *rChannel,
                        unsigned char *gChannel,
                        unsigned char *bChannel,
                        unsigned char *grayImg);                    // Sequentially convert an RGB image to grayscale.

void seqConvolve(unsigned int imgWidth,                     
                            unsigned int imgHeight,
                            unsigned int maskSize,
                            unsigned char *inputImg,
                            float *mask,
                            unsigned char *outputImg);              // Sequentially convolve an image with a filter.

void seqFilter(unsigned int imgWidth,                       
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg);              // Sequentially filter an image.

bool checkEquality(unsigned char* img1, 
                    unsigned char* img2, 
                    const int W, 
                    const int H);                                   // Check if the images img1 and img2 are equal.

void displayImg(unsigned char *img, int imgWidth, int imgHeight);   // Display unsigned char matrix as an image.

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

cl::Device getDefaultDevice();                                    // Return a device found in this OpenCL platform.

void initializeDevice();                                          // Inicialize device and compile kernel code.

void parFilter(unsigned int imgWidth,                       
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg);             // Parallelly filter an image.

// =================================================================
// ------------------------ Global Variables ------------------------
// =================================================================

cl::Program program;                // The program that will run on the device.    
cl::Context context;                // The context which holds the device.    
cl::Device device;                  // The device where the kernel will run.

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main(){

    /**
     * Create auxiliary variables.
     * */

    clock_t start, end;

    /**
     * Load input image.
     * */

    CImg<unsigned char> cimg("input_img.jpg");
    unsigned char *inputImg = cimg.data();
    unsigned int imgWidth = cimg.width(), imgHeight = cimg.height();
    unsigned char *inputRchannel = &inputImg[0];
    unsigned char *inputGchannel = &inputImg[imgWidth*imgHeight];
    unsigned char *inputBchannel = &inputImg[2*imgWidth*imgHeight];

    /**
     * Create a low-pass filter mask.
     * */

    const int lpMaskSize = 5;
    float lpMask[lpMaskSize][lpMaskSize] = {
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
        {.04,.04,.04,.04,.04},
    };
    float* lpMaskData = &lpMask[0][0];

    /**
     * Create a high-pass filter mask.
     * */

    const int hpMaskSize = 5;
    float hpMask[hpMaskSize][hpMaskSize] = {
        {-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1},
        {-1,-1,24,-1,-1},
        {-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1},
    };
    float* hpMaskData = &hpMask[0][0];

    /**
     * Allocate memory for the output images.
     * */

    unsigned char *seqFilteredImg = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    unsigned char *parFilteredImg = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    
    /**
     * Sequentially convolve filter over image.
     * */

    start = clock();
    seqFilter(imgWidth, imgHeight, lpMaskSize, hpMaskSize, inputRchannel, inputGchannel, inputBchannel, 
    lpMaskData, hpMaskData, seqFilteredImg);
    end = clock();
    double seqTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC;

    /**
     * Initialize OpenCL device.
     */

    initializeDevice();

    /**
     * Parallelly convolve filter over image.
     * */
    
    start = clock();
    parFilter(imgWidth, imgHeight, lpMaskSize, hpMaskSize, inputRchannel, inputGchannel, inputBchannel, 
    lpMaskData, hpMaskData, parFilteredImg);
    end = clock();
    double parTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC;
    
    /**
     * Check if outputs are equal.
     * */

    bool equal = checkEquality(seqFilteredImg, parFilteredImg, imgWidth, imgHeight);

    /**
     * Print results.
     */

    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << seqTime << " ms;\n\tParallel: " << parTime << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (seqTime - parTime) / parTime) << "\%\n";

    /**
     * Display filtered image.
     * */
    
    displayImg(parFilteredImg, imgWidth, imgHeight);
    return 0;
}

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

/**
 * Return a device found in this OpenCL platform.
 * */

cl::Device getDefaultDevice(){
    // CODE HERE !!!
}

/**
 * Inicialize device and compile kernel code.
 * */

void initializeDevice(){
    // CODE HERE !!!
}

/**
 * Parallelly filter an image.
 */

void parFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg){
    // CODE HERE !!!
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

/**
 * Sequentially convert an RGB image to grayscale.
 */

void seqRgb2Gray(unsigned int imgWidth,
                        unsigned int imgHeight,
                        unsigned char *rChannel,
                        unsigned char *gChannel,
                        unsigned char *bChannel,
                        unsigned char *grayImg){
    
    /**
     * Declare the current index variable.
     */

    size_t idx;

    /**
     * Loop over input image pixels.
     */

    for(int i = 0; i < imgWidth; i++){
        for(int j = 0; j < imgHeight; j++){

            /**
             * Compute average pixel.
             */

            idx = i + j*imgWidth;
            grayImg[idx] = (rChannel[idx] + gChannel[idx] + bChannel[idx]) / 3;
        }
    }
}

/**
 * Sequentially convolve an image with a filter mask.
 */

void seqConvolve(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int maskSize,
                            unsigned char *inputImg,
                            float *mask,
                            unsigned char *outputImg){
    /**
     * Loop through input image.
     * */

    for(size_t i = 0; i < imgWidth; i++){
        for(size_t j = 0; j < imgHeight; j++){
                
            /**
             * Check if the mask cannot be applied to the
             * current image pixel.
             * */
            
            if(i < maskSize/2  
            || j < maskSize/2
            || i >= imgWidth - maskSize/2
            || j >= imgHeight - maskSize/2){
                outputImg[i + j * imgWidth] = 0;
                continue;
            }
            
            /**
             * Apply mask based on the neighborhood of pixel inputImg(j,i).
             * */
            
            int outSum = 0;
            for(size_t k = 0; k < maskSize; k++){
                for(size_t l = 0; l < maskSize; l++){
                  size_t colIdx = i - maskSize/2 + k;
                  size_t rowIdx = j - maskSize/2 + l;
                  size_t maskIdx = (maskSize-1-k) + (maskSize-1-l)*maskSize;
                  outSum += inputImg[rowIdx * imgWidth + colIdx] * mask[maskIdx];
                }
            }

            /**
             * Update output pixel.
             * */

            if(outSum < 0){
                outputImg[i + j * imgWidth] = 0;
            } else if(outSum > 255){
                outputImg[i + j * imgWidth] = 255;
            } else{
                outputImg[i + j * imgWidth] = outSum;
            }
        }
    }
}

/**
 * Sequentially filter an image.
 */

void seqFilter(unsigned int imgWidth,
                            unsigned int imgHeight,
                            unsigned int lpMaskSize,
                            unsigned int hpMaskSize,
                            unsigned char *inputRchannel,
                            unsigned char *inputGchannel,
                            unsigned char *inputBchannel,
                            float *lpMask,
                            float *hpMask,
                            unsigned char *outputImg){

    /**
     * Convert input image to grayscale.
     */

    unsigned char *grayOut = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    seqRgb2Gray(imgWidth, imgHeight, inputRchannel, inputGchannel, inputBchannel, grayOut);

    /**
     * Apply the low-pass filter.
     */

    unsigned char *lpOut = (unsigned char*) malloc(imgWidth * imgHeight * sizeof(unsigned char));
    seqConvolve(imgWidth, imgHeight, lpMaskSize, grayOut, lpMask, lpOut);
    
    /**
     * Apply the high-pass filter.
     */

    seqConvolve(imgWidth, imgHeight, hpMaskSize, lpOut, hpMask, outputImg);
}

/**
 * Display unsigned char matrix as an image.
 * */

void displayImg(unsigned char *img, int imgWidth, int imgHeight){
    
    /**
     * Create C_IMG object.
     * */
    
    CImg<unsigned char> cimg(imgWidth, imgHeight);

    /**
     * Transfer image data to C_IMG object.
     * */

    for(int i = 0; i < imgWidth; i++){
        for(int j = 0; j < imgHeight; j++){
            cimg(i,j) = img[i + imgWidth*j];
        }
    }

    /**
     * Display image.
     * */

    cimg.display();
}

/**
 * Check if the images img1 and img2 are equal.
 * */

bool checkEquality(unsigned char* img1,
                unsigned char* img2, 
                const int M, 
                const int N){
    for(int i = 0; i < M*N; i++){
        if(img1[i] != img2[i]){
            return false;
        }
    }
    return true;
}
