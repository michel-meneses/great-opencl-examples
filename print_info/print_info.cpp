#include <CL/cl.hpp>
#include <iostream>

int main(){

    /**
     * Search for all the OpenCL platforms available and check
     * if there are any.
     * */

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()){
        std::cerr << "No platforms found!" << std::endl;
        return -1;
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
        return -1;
    }

    /**
     * Select the first device and print its information.
     * */

    auto device = devices.front();
    auto name = device.getInfo<CL_DEVICE_NAME>();
    auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
    auto version = device.getInfo<CL_DEVICE_VERSION>();
    auto workItems = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    auto workGroups = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    auto computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    auto globalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    auto localMemory = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    std::cout << "OpenCL Device Info:"
    << "\nName: " << name
    << "\nVendor: " << vendor
    << "\nVersion: " << version
    << "\nMax size of work-items: (" << workItems[0] << "," << workItems[1] << "," << workItems[2] << ")"
    << "\nMax size of work-groups: " << workGroups
    << "\nNumber of compute units: " << computeUnits
    << "\nGlobal memory size (bytes): " << globalMemory
    << "\nLocal memory size per compute unit (bytes): " << localMemory/computeUnits
    << std::endl;

    return 0;
}