#include<iostream>
#include<tensor.cuh>
#include<device/runtime.cuh>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    const tensor_d a = {1, 2, 3, 4}, b = {5, 6, 7, 8};
    
    const auto result = tensor_d::div(a, b);
    cout << "result = " << result << endl;

    cout << endl;
    return 0;
}           