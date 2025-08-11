#include<cstdio>
#include <initializer_list>
#include<iostream>
#include<tensor.cuh>
#include<init_tensor.h>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    tensor_d a = {{1, 2}, {3, 4}, {6, 7}}, b = {2, 2};
    tensor_d result = a + b;

    cout << "Result = " << result << endl;

    cout << endl;
    return 0;
} 