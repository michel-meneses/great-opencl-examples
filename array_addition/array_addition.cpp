#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

cl::Device getDefaultDevice();                          // Return the first device found in this OpenCL platform.
void initializeDevice();                                // Inicialize device and compile kernel code.
void seqSumArrays(int* a, int* b, int* c, const int N); // Sequentially performs the N-dimensional operation c = a + b.
void parSumArrays(int* a, int* b, int* c, const int N); // Parallelly performs the N-dimensional operation c = a + b.
bool checkEquality(int* c1, int* c2, const int N);      // Check if the N-dimensional arrays c1 and c2 are equal.

// =================================================================
// ------------------------ Global Variables ------------------------
// =================================================================

cl::Program program;    // The program that will run on the device.    
cl::Context context;    // The context which holds the device.    
cl::Device device;      // The device where the kernel will run.

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main(){
    
    /**
     * Create auxiliary variables.
     * */

    clock_t start, end;
    const int EXECUTIONS = 10;

    /**
     * Prepare input arrays.
     * */

    int ARRAYS_DIM = 1 << 20;
    std::vector<int> a(ARRAYS_DIM, 3);
    std::vector<int> b(ARRAYS_DIM, 5);

    /**
     * Prepare sequential and parallel outputs.
     * */

    std::vector<int> cs(ARRAYS_DIM);
    std::vector<int> cp(ARRAYS_DIM);

    /**
     * Sequentially sum arrays.
     * */

    start = clock();
    for(int i = 0; i < EXECUTIONS; i++){
        seqSumArrays(a.data(), b.data(), cs.data(), ARRAYS_DIM);
    }
    end = clock();
    double seqTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

    /**
     * Initialize OpenCL device.
     * */

    initializeDevice();

    /**
     * Parallelly sum arrays.
     * */

    start = clock();
    for(int i = 0; i < EXECUTIONS; i++){
        parSumArrays(a.data(), b.data(), cp.data(), ARRAYS_DIM);
    }
    end = clock();
    double parTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

    /**
     * Check if outputs are equal.
     * */

    bool equal = checkEquality(cs.data(), cp.data(), ARRAYS_DIM);

    /**
     * Print results.
     * */

    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0] << "\n\tc[0] = a[0] + b[0] = " << cp[0] << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << seqTime << " ms;\n\tParallel: " << parTime << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (seqTime - parTime) / parTime) << "\%\n";
    return 0;
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

/**
 * Return the first device found in this OpenCL platform.
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
     * Search for all the devices on the first platform and check if
     * there are any available.
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

    std::ifstream kernel_file("array_addition.cl");
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
 * Sequentially performs the N-dimensional operation c = a + b.
 * */

void seqSumArrays(int* a, int* b, int* c, const int N){
    for(int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}

/**
 * Parallelly performs the N-dimensional operation c = a + b.
 * */

void parSumArrays(int* a, int* b, int* c, const int N){
    
    /**
     * Create buffers and allocate memory on the device.
     * */

    cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * sizeof(int), a);
    cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, N * sizeof(int), b);
    cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * sizeof(int));

    /**
     * Set kernel arguments.
     * */

    cl::Kernel kernel(program, "sumArrays");
    kernel.setArg(0, aBuf);
    kernel.setArg(1, bBuf);
    kernel.setArg(2, cBuf);

    /**
     * Execute the kernel function and collect its result.
     * */

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N));
    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, N * sizeof(int), c);
}

/**
 * Check if the N-dimensional arrays c1 and c2 are equal.
 * */

bool checkEquality(int* c1, int* c2, const int N){
    for(int i = 0; i < N; i++){
        if(c1[i] != c2[i]){
            return false;
        }
    }
    return true;
}
