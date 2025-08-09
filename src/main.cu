#include<cstdio>
#include<iostream>
#include<tensor.cuh>
#include<init_tensor.h>

using std::cout;
using std::endl;
using tensor_d = tensor<double>;

int main(void) {
    cout << endl;

    tensor_d inputs = {
        {  1.0,  2.0,  3.0,  2.5 },
        {  2.0,  5.0, -1.0,  2.0 },
        { -1.5,  2.7,  3.3, -0.8 }
    };

    tensor_d weights = {
        {  0.2,   0.8,  -0.5,  1.0  },
        {  0.5,  -0.91,  0.26, -0.5 },
        { -0.26, -0.27,  0.17, 0.87 }
    };

    tensor_d biases = { 2.0, 3.0, 0.5 };

    auto layer1_outputs = inputs * tensor_d::transpose(weights) + biases;

    weights = {
        {  0.1,  -0.14,  0.5  },
        { -0.5,   0.12, -0.33 },
        { -0.44,  0.73, -0.13 }
    };

    biases = { -1.0, 2.0, -0.5 };    

    auto layer2_outputs = layer1_outputs * tensor_d::transpose(weights) + biases; 

    cout << "output = " << layer2_outputs << endl;

    cout << endl;
    return 0;
} 