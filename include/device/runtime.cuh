#pragma once

#include<device/constants.h>

namespace device_runtime {
    extern __device__ unsigned int error_code;
    __device__ inline void set_error(device_constants::error code) { 
        atomicExch(&error_code, static_cast<int>(code)); 
    }
}