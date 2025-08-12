#include "host/runtime.h"
#include<iostream>
#include<tensor.cuh>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    const tensor_d a = {1, 2, 3 ,4}, b = {0, 6, 7, 8};
    
    try {
        const auto result = tensor_d::div(a, b);
        cout << "result = " << result << endl;
    } catch (host_runtime::device_exception& e) {
        e.what();
    }

    cout << endl;
    return 0;
} 