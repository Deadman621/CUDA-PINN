#include<cstdio>
#include<iostream>
#include<tensor.cuh>
#include<init_tensor.h>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    tensor_d a = {1, 2, 3, 4}, b = {2};
    b(0) = 5;
    a(0) = 13;

    auto result = a + b;

    cout << "result = " << result << endl;

    cout << endl;
    return 0;
} 