#include <torch/extension.h>

#include <vector>

// cuda 

torch::Tensor segment_mm_cuda_forward(
    const torch::Tensor& mat_A, 
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A,
    const torch::Tensor& segment_id_B);

std::vector<torch::Tensor> segment_mm_cuda_backward(
    const torch::Tensor& grad_c,
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A, 
    const torch::Tensor& segment_id_B);


// c++

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor segment_mm_forward(
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A,
    const torch::Tensor& segment_id_B) {
    CHECK_INPUT(mat_A);
    CHECK_INPUT(mat_B);
    CHECK_INPUT(segment_id_A);
    CHECK_INPUT(segment_id_B);
    
    return segment_mm_cuda_forward(mat_A, mat_B, segment_id_A, segment_id_B);
}


std::vector<torch::Tensor> segment_mm_backward(
    const torch::Tensor& grad_c,
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A, 
    const torch::Tensor& segment_id_B) {
    CHECK_INPUT(grad_c);
    CHECK_INPUT(mat_A);
    CHECK_INPUT(mat_B);
    CHECK_INPUT(segment_id_A);
    CHECK_INPUT(segment_id_B);

    return segment_mm_cuda_backward(grad_c, mat_A, mat_B, segment_id_A, segment_id_B);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &segment_mm_forward, "segment_mm forward (CUDA)");
    m.def("backward", &segment_mm_backward, "segment_mm backward (CUDA)");
}