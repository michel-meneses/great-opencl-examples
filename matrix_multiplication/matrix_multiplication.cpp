#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

cl::Device getDefaultDevice();        // Return the first device found in this OpenCL platform.
void initializeDevice();              // Inicialize device and compile kernel code.
void seqMultiplyMatrices(int* a, 
                        int* b, 
                        int* c, 
                        const int M, 
                        const int N, 
                        const int K); // Sequentially performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatrices(int* a, 
                        int* b, 
                        int* c, 
                        const int M, 
                        const int N, 
                        const int K); // Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
bool checkEquality(int* c1, 
                    int* c2, 
                    const int M, 
                    const int N);     // Check if the matrices c1 and c2 are equal.

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
    const int EXECUTIONS = 40;
    
    /**
     * Prepare input constants related to the dimensions of the matrices.
     * */

    const int M = 1 << 4;
    const int N = 1 << 4;
    const int K = 1 << 12;

    /**
     * Prepare input matrices A and B.
     * */

    const size_t ROWS_A = M;
    const size_t COLS_A = K;
    std::vector<int> a(ROWS_A * COLS_A, 3);

    const size_t ROWS_B = K;
    const size_t COLS_B = N;
    std::vector<int> b(ROWS_B * COLS_B, 5);

    /**
     * Prepare sequential and parallel output matrices.
     * */

    const size_t ROWS_C = M;
    const size_t COLS_C = N;
    std::vector<int> cs(ROWS_C * COLS_C);
    std::vector<int> cp(ROWS_C * COLS_C);

    /**
     * Sequentially multiply matrices.
     * */

    start = clock();
    for(int i = 0; i < EXECUTIONS; i++){
        seqMultiplyMatrices(a.data(), b.data(), cs.data(), M, N, K);
    }
    end = clock();
    double seqTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

    /**
     * Initialize OpenCL device.
     * */

    initializeDevice();

    /**
     * Parallelly multiply matrices.
     * */

    start = clock();
    for(int i = 0; i < EXECUTIONS; i++){
        parMultiplyMatrices(a.data(), b.data(), cp.data(), M, N, K);
    }
    end = clock();
    double parTime = ((double) 10e3 * (end - start)) / CLOCKS_PER_SEC / EXECUTIONS;

    /**
     * Check if outputs are equal.
     * */

    bool equal = checkEquality(cs.data(), cp.data(), ROWS_C, COLS_C);

    /**
     * Print results.
     * */

    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Results: \n\tA[0] = " << a[0] << "\n\tB[0] = " << b[0] << "\n\tC[0] = " << cp[0] << std::endl;
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

    std::ifstream kernel_file("matrix_multiplication.cl");
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
 * Sequentially performs the operation c[M,N] = a[M,K] * b[K,N].
 * */

void seqMultiplyMatrices(int* a, int* b, int* c, 
                         const int M, 
                         const int N, 
                         const int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            int sum = 0;
            for(int k = 0; k < K; k++){
                sum += a[i*K + k] * b[j + k*N];
            }
            c[i*N + j] = sum;
        }
    }
}

/**
 * Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
 * */

void parMultiplyMatrices(int* a, int* b, int* c, 
                        const int M, 
                        const int N,
                        const int K){
    
    /**
     * Create buffers and allocate memory on the device.
     * */

    cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, M * K * sizeof(int), a);
    cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, K * N * sizeof(int), b);
    cl::Buffer cBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));

    /**
     * Set kernel arguments.
     * */

    cl::Kernel kernel(program, "multiplyMatrices");
    kernel.setArg(0, aBuf);
    kernel.setArg(1, bBuf);
    kernel.setArg(2, cBuf);
    kernel.setArg(3, sizeof(unsigned int), &M);
    kernel.setArg(4, sizeof(unsigned int), &N);
    kernel.setArg(5, sizeof(unsigned int), &K);

    /**
     * Execute the kernel function and collect its result.
     * */

    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, M));
    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
    queue.finish();
}

/**
 * Check if the matrices C1 and C2 are equal.
 * */

bool checkEquality(int* c1, int* c2, 
                  const int M, 
                  const int N){
    for(int i = 0; i < M*N; i++){
        if(c1[i] != c2[i]){
            return false;
        }
    }
    return true;
}
