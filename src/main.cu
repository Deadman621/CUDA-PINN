#include<iostream>
#include<utils.cuh>
#include<vector>

using namespace std;

int main(void) {

    cout << "PINN" << endl;
    tensor<float> t1({1, 2, 3}), t2({4, 5, 6}), t3({7, 8, 9});
    tensor<float> vec_add = t1 + t2 + t3;
    float* x = vec_add.raw();
    cout << "vec_add: " << "[";
    for(size_t i = 0; i < vec_add.size(); i++)
        cout << x[i] << ", ";
    cout << "\b\b]" << endl;

    std::vector<int> v = {1, 2, 3};
    auto a = v[0];  // <-- hover here

    return 0;
}   