#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// hyper parameter 
constexpr int BLOCK_SIZE = 16; 

namespace {

template <typename scalar_t> 
__global__ void tiled_mm_kernel(
    const scalar_t* __restrict__ a, 
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    int offset, int N, int M, int D) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    float temp = 0.0f;
    int row = threadIdx.x;
    int col = threadIdx.y;

    #pragma unroll
    for (int k = 0; k <= (D / BLOCK_SIZE); ++k) {
        // load 
        if ((BLOCK_SIZE*k + col) >= D)
            As[row][col] = 0.0f;
        else 
            As[row][col] = a[D*BLOCK_SIZE*block_row + BLOCK_SIZE*k + row*D + col];
        
        if ((BLOCK_SIZE*k + row) >= D)
            As[row][col] = 0.0f;
        else 
            As[row][col] = b[M*BLOCK_SIZE*k + BLOCK_SIZE*block_col + row*M + col];

        __syncthreads();
        #pragma unroll
        for (int d = 0; d < BLOCK_SIZE; ++d)
            temp += As[row][d]*Bs[d][col];
        
        __syncthreads();
    }

    if ((BLOCK_SIZE*block_col + col) < M && (BLOCK_SIZE*block_row + row) < N)
        c[M*BLOCK_SIZE*block_row + BLOCK_SIZE*block_col + row*M + col + offset] = temp;
}

}


torch::Tensor segment_mm_cuda_forward(
    const torch::Tensor& mat_A, 
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A,
    const torch::Tensor& segment_id_B) {
    const int D = mat_A.size(1);
    const int N = mat_A.size(0);
    const int M = mat_B.size(0);
    
    cudaSetDevice(mat_A.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto count_A = segment_id_A.bincount();
    auto count_B = segment_id_B.bincount();
    
    // allocate C
    int sum = 0;
    const int size = count_A.size(0);
    for (int i = 0; i < size; ++i) {
        auto N_i = count_A[i].item().toInt();
        auto M_i = count_B[i].item().toInt();

        sum += N_i * M_i;
    }
    auto C = torch::zeros({sum});

    // loop k times, launch k kernels 
    int start_A = 0, start_B = 0, offset = 0;    
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    for (int i = 0; i < size; ++i) {
        auto N_i = count_A[i].item().toInt();
        auto M_i = count_B[i].item().toInt();
        
        auto A_i = mat_A.narrow(0, start_A, N_i); // [N_i, D]
        auto B_i = mat_B.narrow(0, start_B, M_i).transpose(0, 1); // [D, M_i] 
        
        dim3 dim_grid((N_i + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (M_i + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // async dispatch
        AT_DISPATCH_FLOATING_TYPES(mat_A.type(), "segment_mm_cuda_forward", ([&]{
           tiled_mm_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
               A_i.data<scalar_t>(),
               B_i.data<scalar_t>(),
               C.data<scalar_t>(),
               offset, N_i, M_i, D);
        }));

        start_A += N_i;
        start_B += M_i;
        offset += N_i * M_i;
    }

    return C;
}

std::vector<torch::Tensor> segment_mm_cuda_backward(
    const torch::Tensor& grad_c,
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A, 
    const torch::Tensor& segment_id_B) {
    
}
