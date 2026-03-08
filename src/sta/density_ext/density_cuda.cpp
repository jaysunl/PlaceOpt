#include <torch/extension.h>
#include <vector>

torch::Tensor density_forward_cuda(
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size);

std::vector<torch::Tensor> density_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size);

static torch::Tensor density_forward(
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size) {
    return density_forward_cuda(boundary, xy, wh, weight, weight_scalar, weight_is_tensor, grid_size);
}

static std::vector<torch::Tensor> density_backward(
    torch::Tensor grad_out,
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size) {
    return density_backward_cuda(grad_out, boundary, xy, wh, weight, weight_scalar, weight_is_tensor, grid_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &density_forward, "Density forward (CUDA)");
    m.def("backward", &density_backward, "Density backward (CUDA)");
}
