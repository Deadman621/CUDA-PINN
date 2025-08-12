#pragma once

#include<device/constants.h>

namespace device_runtime {
    // NEEDS EXTERNAL LINKAGE - USING INTERNAL FOR NOW
    __device__ static unsigned int error_code = device_constants::error::NORMAL;
    __device__ inline void set_error(device_constants::error code) { 
        atomicExch(&error_code, static_cast<int>(code)); 
    }
}