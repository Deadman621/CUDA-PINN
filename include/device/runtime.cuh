#pragma once

#include<device/constants.h>
#include<string_view>
#include<string>
#include<atomic>

namespace device {
    namespace runtime {
        // NEEDS EXTERNAL LINKAGE - USING INTERNAL FOR NOW
        __device__ static unsigned int d_error_code = constants::error::NORMAL;
        __device__ inline void set_error(constants::error code) { 
            atomicExch(&d_error_code, static_cast<int>(code)); 
        }

        inline std::atomic<constants::error> error_code = constants::error::NORMAL;
        
        constexpr std::string_view make_string(constants::error code) {
            switch(code) {
                case constants::error::NORMAL:           return "success";
                case constants::error::DIVISION_BY_ZERO: return "division by zero";
                case constants::error::DEVICE_OVERFLOW:  return "overflow"; 
                default:                                 return "unknown error";
            }
        }
        
        class device_exception: public std::exception {
            private:
                constants::error code_;
                std::string message_;

            public:
                explicit device_exception(constants::error code, const std::string_view& name): code_{code} {
                    this->message_ = std::string{name} + ": " + std::string{make_string(code)};
                }

                const char *what(void) const noexcept override { return this->message_.data(); }
                constants::error code(void) const noexcept { return this->code_; }
        };

        inline void find_error(const std::string& name) {
            int temp = 0;
            cudaMemcpyFromSymbol(&temp, d_error_code, sizeof(int), 0, cudaMemcpyDeviceToHost);
            error_code.store(static_cast<constants::error>(temp));
            if (error_code.load() == constants::error::NORMAL) return;

            throw device_exception{error_code.load(), name};
        }
    }
}