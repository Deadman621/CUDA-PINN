#include<cstdio>
#include<iostream>
#include<memory>
#include<tensor.cuh>
#include<init_tensor.h>

using namespace std;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    std::unique_ptr<tensor_d> ptr(new tensor_d({{1, 2}, {3, 4}}));
    auto result = tensor_d::matmul(*ptr, tensor_d({{5, 6}, {7, 8}}));
    std::cout << "Result = " << result << std::endl;

    auto raw = ptr->raw();

    cout << endl;
    return 0;
} 