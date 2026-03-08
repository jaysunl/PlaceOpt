#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

__global__ void density_forward_kernel(
    const float* boundary,
    const float* xy,
    const float* wh,
    const float* weight,
    float weight_scalar,
    bool weight_is_tensor,
    bool wh_broadcast,
    int64_t n,
    int grid_size,
    float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float xmin = boundary[0];
    float ymin = boundary[1];
    float xmax = boundary[2];
    float ymax = boundary[3];
    float grid_lx = (xmax - xmin) / static_cast<float>(grid_size);
    float grid_ly = (ymax - ymin) / static_cast<float>(grid_size);
    if (grid_lx <= 0.0f || grid_ly <= 0.0f) {
        return;
    }

    float x0 = xy[idx * 2];
    float y0 = xy[idx * 2 + 1];
    float w = wh_broadcast ? wh[0] : wh[idx * 2];
    float h = wh_broadcast ? wh[1] : wh[idx * 2 + 1];
    float x1 = x0 + w;
    float y1 = y0 + h;

    float x0i = fmaxf(x0, xmin);
    float x1i = fminf(x1, xmax);
    float y0i = fmaxf(y0, ymin);
    float y1i = fminf(y1, ymax);
    if (x1i <= x0i || y1i <= y0i) {
        return;
    }

    int ix0 = static_cast<int>(floorf((x0i - xmin) / grid_lx));
    int ix1 = static_cast<int>(floorf((x1i - xmin) / grid_lx));
    int iy0 = static_cast<int>(floorf((y0i - ymin) / grid_ly));
    int iy1 = static_cast<int>(floorf((y1i - ymin) / grid_ly));

    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= grid_size) ix1 = grid_size - 1;
    if (iy1 >= grid_size) iy1 = grid_size - 1;

    float wval = weight_is_tensor ? weight[idx] : weight_scalar;
    float inv_area = 1.0f / (grid_lx * grid_ly);

    for (int ix = ix0; ix <= ix1; ++ix) {
        float gx0 = xmin + static_cast<float>(ix) * grid_lx;
        float gx1 = gx0 + grid_lx;
        float ovx = fminf(gx1, x1) - fmaxf(gx0, x0);
        if (ovx <= 0.0f) {
            continue;
        }
        for (int iy = iy0; iy <= iy1; ++iy) {
            float gy0 = ymin + static_cast<float>(iy) * grid_ly;
            float gy1 = gy0 + grid_ly;
            float ovy = fminf(gy1, y1) - fmaxf(gy0, y0);
            if (ovy <= 0.0f) {
                continue;
            }
            float contrib = wval * ovx * ovy * inv_area;
            atomicAdd(&out[ix * grid_size + iy], contrib);
        }
    }
}

__global__ void density_backward_kernel(
    const float* boundary,
    const float* xy,
    const float* wh,
    const float* weight,
    const float* grad_out,
    float* grad_xy,
    float* grad_wh,
    float* grad_weight,
    float weight_scalar,
    bool weight_is_tensor,
    bool wh_broadcast,
    int64_t n,
    int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float xmin = boundary[0];
    float ymin = boundary[1];
    float xmax = boundary[2];
    float ymax = boundary[3];
    float grid_lx = (xmax - xmin) / static_cast<float>(grid_size);
    float grid_ly = (ymax - ymin) / static_cast<float>(grid_size);
    if (grid_lx <= 0.0f || grid_ly <= 0.0f) {
        return;
    }

    float x0 = xy[idx * 2];
    float y0 = xy[idx * 2 + 1];
    float w = wh_broadcast ? wh[0] : wh[idx * 2];
    float h = wh_broadcast ? wh[1] : wh[idx * 2 + 1];
    float x1 = x0 + w;
    float y1 = y0 + h;

    float x0i = fmaxf(x0, xmin);
    float x1i = fminf(x1, xmax);
    float y0i = fmaxf(y0, ymin);
    float y1i = fminf(y1, ymax);
    if (x1i <= x0i || y1i <= y0i) {
        return;
    }

    int ix0 = static_cast<int>(floorf((x0i - xmin) / grid_lx));
    int ix1 = static_cast<int>(floorf((x1i - xmin) / grid_lx));
    int iy0 = static_cast<int>(floorf((y0i - ymin) / grid_ly));
    int iy1 = static_cast<int>(floorf((y1i - ymin) / grid_ly));

    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= grid_size) ix1 = grid_size - 1;
    if (iy1 >= grid_size) iy1 = grid_size - 1;

    float wval = weight_is_tensor ? weight[idx] : weight_scalar;
    float inv_area = 1.0f / (grid_lx * grid_ly);

    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_w = 0.0f;
    float grad_h = 0.0f;
    float grad_wt = 0.0f;

    for (int ix = ix0; ix <= ix1; ++ix) {
        float gx0 = xmin + static_cast<float>(ix) * grid_lx;
        float gx1 = gx0 + grid_lx;
        float ovx = fminf(gx1, x1) - fmaxf(gx0, x0);
        if (ovx <= 0.0f) {
            continue;
        }
        float d_ovx_dx1 = (x1 < gx1) ? 1.0f : 0.0f;
        float d_ovx_dx0 = (x0 > gx0) ? -1.0f : 0.0f;
        for (int iy = iy0; iy <= iy1; ++iy) {
            float gy0 = ymin + static_cast<float>(iy) * grid_ly;
            float gy1 = gy0 + grid_ly;
            float ovy = fminf(gy1, y1) - fmaxf(gy0, y0);
            if (ovy <= 0.0f) {
                continue;
            }
            float d_ovy_dy1 = (y1 < gy1) ? 1.0f : 0.0f;
            float d_ovy_dy0 = (y0 > gy0) ? -1.0f : 0.0f;
            float go = grad_out[ix * grid_size + iy];
            float base = go * wval * inv_area;
            grad_x += base * ovy * (d_ovx_dx0 + d_ovx_dx1);
            grad_w += base * ovy * d_ovx_dx1;
            grad_y += base * ovx * (d_ovy_dy0 + d_ovy_dy1);
            grad_h += base * ovx * d_ovy_dy1;
            if (weight_is_tensor) {
                grad_wt += go * ovx * ovy * inv_area;
            }
        }
    }

    grad_xy[idx * 2] = grad_x;
    grad_xy[idx * 2 + 1] = grad_y;

    if (wh_broadcast) {
        atomicAdd(&grad_wh[0], grad_w);
        atomicAdd(&grad_wh[1], grad_h);
    } else {
        grad_wh[idx * 2] = grad_w;
        grad_wh[idx * 2 + 1] = grad_h;
    }

    if (weight_is_tensor && grad_weight != nullptr) {
        grad_weight[idx] = grad_wt;
    }
}

static void check_inputs(const torch::Tensor& boundary,
                         const torch::Tensor& xy,
                         const torch::Tensor& wh,
                         const torch::Tensor& weight,
                         bool weight_is_tensor) {
    CHECK_CUDA(boundary);
    CHECK_CUDA(xy);
    CHECK_CUDA(wh);
    if (weight_is_tensor) {
        CHECK_CUDA(weight);
    }
    CHECK_CONTIGUOUS(boundary);
    CHECK_CONTIGUOUS(xy);
    CHECK_CONTIGUOUS(wh);
    if (weight_is_tensor) {
        CHECK_CONTIGUOUS(weight);
    }
    CHECK_FLOAT(boundary);
    CHECK_FLOAT(xy);
    CHECK_FLOAT(wh);
    if (weight_is_tensor) {
        CHECK_FLOAT(weight);
    }
}

torch::Tensor density_forward_cuda(
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size) {
    boundary = boundary.contiguous();
    xy = xy.contiguous();
    wh = wh.contiguous();
    if (weight_is_tensor) {
        weight = weight.contiguous();
    }
    check_inputs(boundary, xy, wh, weight, weight_is_tensor);

    auto out = torch::zeros({grid_size, grid_size}, xy.options());
    if (xy.numel() == 0) {
        return out;
    }

    bool wh_broadcast = (wh.numel() == 2);
    int64_t n = xy.size(0);

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    density_forward_kernel<<<blocks, threads, 0, stream>>>(
        boundary.data_ptr<float>(),
        xy.data_ptr<float>(),
        wh.data_ptr<float>(),
        weight_is_tensor ? weight.data_ptr<float>() : nullptr,
        static_cast<float>(weight_scalar),
        weight_is_tensor,
        wh_broadcast,
        n,
        static_cast<int>(grid_size),
        out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> density_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor boundary,
    torch::Tensor xy,
    torch::Tensor wh,
    torch::Tensor weight,
    double weight_scalar,
    bool weight_is_tensor,
    int64_t grid_size) {
    boundary = boundary.contiguous();
    xy = xy.contiguous();
    wh = wh.contiguous();
    grad_out = grad_out.contiguous();
    if (weight_is_tensor) {
        weight = weight.contiguous();
    }
    check_inputs(boundary, xy, wh, weight, weight_is_tensor);

    auto grad_xy = torch::zeros_like(xy);
    auto grad_wh = torch::zeros_like(wh);
    torch::Tensor grad_weight;
    if (weight_is_tensor) {
        grad_weight = torch::zeros_like(weight);
    } else {
        grad_weight = torch::Tensor();
    }

    if (xy.numel() == 0) {
        return {grad_xy, grad_wh, grad_weight};
    }

    bool wh_broadcast = (wh.numel() == 2);
    int64_t n = xy.size(0);
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    density_backward_kernel<<<blocks, threads, 0, stream>>>(
        boundary.data_ptr<float>(),
        xy.data_ptr<float>(),
        wh.data_ptr<float>(),
        weight_is_tensor ? weight.data_ptr<float>() : nullptr,
        grad_out.data_ptr<float>(),
        grad_xy.data_ptr<float>(),
        grad_wh.data_ptr<float>(),
        weight_is_tensor ? grad_weight.data_ptr<float>() : nullptr,
        static_cast<float>(weight_scalar),
        weight_is_tensor,
        wh_broadcast,
        n,
        static_cast<int>(grid_size));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {grad_xy, grad_wh, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &density_forward_cuda, "Density forward (CUDA)");
    m.def("backward", &density_backward_cuda, "Density backward (CUDA)");
}
