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
                 unsigned char *grayImg);                          // Sequentially convert an RGB image to grayscale.

void seqConvolve(unsigned int imgWidth,                     
                 unsigned int imgHeight,
                 unsigned int maskSize,
                 unsigned char *inputImg,
                 float *mask,
                 unsigned char *outputImg);                        // Sequentially convolve an image with a filter.

void seqFilter(unsigned int imgWidth,                       
               unsigned int imgHeight,
               unsigned int lpMaskSize,
               unsigned int hpMaskSize,
               unsigned char *inputRchannel,
               unsigned char *inputGchannel,
               unsigned char *inputBchannel,
               float *lpMask,
               float *hpMask,
               unsigned char *outputImg);                           // Sequentially filter an image.

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
               unsigned char *outputImg);                        // Parallelly filter an image.

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
    
    /**
     * Search for all the OpenCL platforms available and check
     * if there are any.
     * */

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()){
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    }

    /**
     * Search for all the devices on the first platform 
     * and check if there are any available.
     * */

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()){
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    /**
     * Return the first device found.
     * */

    return devices.front();
}

/**
 * Inicialize device and compile kernel code.
 * */

void initializeDevice(){

    /**
     * Select the first available device.
     * */

    device = getDefaultDevice();
    
    /**
     * Read OpenCL kernel file as a string.
     * */

    std::ifstream kernel_file("image_filtering.cl");
    std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    /**
     * Compile kernel program which will run on the device.
     * */

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    context = cl::Context(device);
    program = cl::Program(context, sources);
    
    auto err = program.build();
    if(err != CL_BUILD_SUCCESS){
        std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) 
        << "\nBuild Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
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
    
    /**
     * Create buffers and allocate memory on the device.
     * */

    cl::Buffer inputRchannelBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * sizeof(unsigned char), inputRchannel);
    cl::Buffer inputGchannelBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * sizeof(unsigned char), inputGchannel);
    cl::Buffer inputBchannelBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, imgWidth * imgHeight * sizeof(unsigned char), inputBchannel);
    cl::Buffer grayOutputBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, imgWidth * imgHeight * sizeof(unsigned char));
    cl::Buffer lpMaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, lpMaskSize * lpMaskSize * sizeof(float), lpMask);
    cl::Buffer hpMaskBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, hpMaskSize * hpMaskSize * sizeof(float), hpMask);
    cl::Buffer lpOutputBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, imgWidth * imgHeight * sizeof(unsigned char));
    cl::Buffer hpOutputBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgWidth * imgHeight * sizeof(unsigned char));

    /**
     * Initialize grayscale kernel.
     * */

    cl::Kernel grayKernel(program, "rgb2gray");
    grayKernel.setArg(0, inputRchannelBuf);
    grayKernel.setArg(1, inputGchannelBuf);
    grayKernel.setArg(2, inputBchannelBuf);
    grayKernel.setArg(3, grayOutputBuf);

    /**
     * Initialize low-pass filter kernel.
     * */
    cl::Kernel lpKernel(program, "filterImageWithCache");
    lpKernel.setArg(0, sizeof(unsigned int), &lpMaskSize);
    lpKernel.setArg(1, grayOutputBuf);
    lpKernel.setArg(2, lpMaskBuf);
    lpKernel.setArg(3, lpOutputBuf);

    /**
     * Initialize high-pass filter kernel.
     * */

    cl::Kernel hpKernel(program, "filterImageWithCache");
    hpKernel.setArg(0, sizeof(unsigned int), &hpMaskSize);
    hpKernel.setArg(1, lpOutputBuf);
    hpKernel.setArg(2, hpMaskBuf);
    hpKernel.setArg(3, hpOutputBuf);

    /**
     * Execute kernel functions and collect the final result.
     * */

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(grayKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight));
    queue.enqueueNDRangeKernel(lpKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight), cl::NDRange(16, 16));
    queue.enqueueNDRangeKernel(hpKernel, cl::NullRange, cl::NDRange(imgWidth, imgHeight), cl::NDRange(16, 16));
    queue.enqueueReadBuffer(hpOutputBuf, CL_TRUE, 0, imgWidth * imgHeight * sizeof(unsigned char), outputImg);
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
