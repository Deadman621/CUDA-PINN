#pragma once

#include<atomic>
#include<string>
#include<exception>
#include<string_view>
#include<device/constants.h>
#include<device/runtime.cuh>

namespace host_runtime {
    using device_constants::error;

    inline std::atomic<device_constants::error> error_code = error::NORMAL;
    
    constexpr std::string_view make_string(device_constants::error code) {
        switch(code) {
            case error::NORMAL:           return "success";
            case error::DIVISION_BY_ZERO: return "division by zero";
            case error::DEVICE_OVERFLOW:  return "overflow"; 
            default:                      return "unknown error";
        }
    }
    
    class device_exception: public std::exception {
        private:
            device_constants::error code_;
            std::string message_;

        public:
            explicit device_exception(error code, const std::string_view& name): code_{code} {
                this->message_ = std::string{name} + ": " + std::string{make_string(code)};
            }

            const char *what(void) const noexcept override { return this->message_.data(); }
            device_constants::error code(void) const noexcept { return this->code_; }
    };

    inline void find_error(const std::string& name) {
        int temp = 0;
        cudaMemcpyFromSymbol(&temp, device_runtime::error_code, sizeof(int), 0, cudaMemcpyDeviceToHost);
        error_code.store(static_cast<device_constants::error>(temp));
        if (error_code.load() == error::NORMAL) return;

        throw device_exception{error_code.load(), name};
    }
}