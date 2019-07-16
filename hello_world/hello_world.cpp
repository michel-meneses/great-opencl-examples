#include <CL/cl.hpp>
#include <fstream>
#include <iostream>

cl::Device get_default_device(){
    
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

int main(){

    /**
     * Select a device.
     * */

    auto device = get_default_device();

    /**
     * Read OpenCL kernel file as a string.
     * */

    std::ifstream hello_world_file("hello_world.cl");
    std::string src(std::istreambuf_iterator<char>(hello_world_file), (std::istreambuf_iterator<char>()));

    /**
     * Compile the program which will run on the device.
     * */

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
    cl::Context context(device);
    cl::Program program(context, sources);

    auto err = program.build();
    if(err != CL_BUILD_SUCCESS){
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) 
        << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }
    
    /**
     * Create buffers and allocate memory on the device.
     * */

    char buf[16];
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "helloWorld", nullptr);

    /**
     * Set kernel argument.
     * */

    kernel.setArg(0, memBuf);

    /**
     * Run the kernel function and collect its result.
     * */

    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buf), buf);

    /**
     * Print result.
     * */

    std::cout << buf;
    return 0;
}