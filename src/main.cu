#include<iostream>
#include<tensor.cuh>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    tensor_d a = {{1, 2}, {3, 4}, {6, 7}}, b = {2, 2};

    tensor_d add = a + b;
    tensor_d sub = a - b;
    tensor_d mul = a * b;
    tensor_d div = a / b;

    cout << "add = " << add << endl;
    cout << "sub = " << sub << endl;
    cout << "mul = " << mul << endl;
    cout << "div = " << div << endl;

    cout << endl;
    return 0;
} 