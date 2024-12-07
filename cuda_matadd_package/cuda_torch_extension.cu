#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int N = a.numel();
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
}

// PyBind11 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Matrix addition using CUDA");
}
