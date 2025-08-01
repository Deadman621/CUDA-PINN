#include<iostream>
#include<vector>
#include<initializer_list>
#include<tensor.cuh>

using namespace std;

int main(void) {
    cout << endl;
    using tensor_d = tensor<double>;

    tensor_d inputs = {{1.0, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}};
    tensor_d weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    tensor_d biases = {{2.0, 3.0, 0.5}, {2.0, 3.0, 0.5}, {2.0, 3.0, 0.5}};

    auto layer1_outputs = inputs * tensor_d::transpose(weights) + biases;

    weights = {{0.1, -0.14, 0.5}, {-0.5, 0.12, -0.33}, {-0.44, 0.73, -0.13}};
    biases = {{-1, 2, -0.5}, {-1, 2, -0.5}, {-1, 2, -0.5}};

    auto layer2_outputs = layer1_outputs * tensor_d::transpose(weights) + biases;

    std::cout << "layer 2 = " << layer2_outputs << std::endl;

    try { cout << "inputs[2][4] = " << inputs(2, 4) << endl; }
    catch (out_of_range e) { cout << e.what() << endl; }

    cout << endl;
    return 0;
} 