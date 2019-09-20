#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include <vector>

// hyper parameter 
constexpr long BLOCK_SIZE = 16; 

namespace {

template <typename scalar_t> 
__global__ void tiled_mm_kernel(
    const scalar_t* __restrict__ a, 
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    long N, long M, long D) {
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    long block_row = blockIdx.x;
    long block_col = blockIdx.y;

    scalar_t temp = 0.0f;
    long row = threadIdx.x;
    long col = threadIdx.y;

    #pragma unroll
    for (long k = 0; k <= (D / BLOCK_SIZE); ++k) {
        // load 
        if ((BLOCK_SIZE*k + col) >= D)
            As[row][col] = 0.0f;
        else 
            As[row][col] = a[D*BLOCK_SIZE*block_row + BLOCK_SIZE*k + row*D + col];
        
        if ((BLOCK_SIZE*k + row) >= D)
            Bs[row][col] = 0.0f;
        else 
            Bs[row][col] = b[M*BLOCK_SIZE*k + BLOCK_SIZE*block_col + row*M + col];

        __syncthreads();
        #pragma unroll
        for (long d = 0; d < BLOCK_SIZE; ++d)
            temp += As[row][d]*Bs[d][col];
        
        __syncthreads();
    }

    if ((BLOCK_SIZE*block_col + col) < M && (BLOCK_SIZE*block_row + row) < N)
        c[M*BLOCK_SIZE*block_row + BLOCK_SIZE*block_col + row*M + col] = temp;
}

// a should be tranposed
template <typename scalar_t> 
__global__ void tiled_mm_kernel_with_transpose_1(
    const scalar_t* __restrict__ a, 
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    long N, long M, long D) {
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    long block_row = blockIdx.x;
    long block_col = blockIdx.y;

    scalar_t temp = 0.0f;
    long row = threadIdx.x;
    long col = threadIdx.y;

    #pragma unroll
    for (long k = 0; k <= (D / BLOCK_SIZE); ++k) {
        // load 
        if ((BLOCK_SIZE*k + row) >= D)
            As[col][row] = 0.0f;
        else 
            As[col][row] = a[M*BLOCK_SIZE*k + BLOCK_SIZE*block_col + row*M + col];
        
        if ((BLOCK_SIZE*k + row) >= D)
            Bs[row][col] = 0.0f;
        else 
            Bs[row][col] = b[M*BLOCK_SIZE*k + BLOCK_SIZE*block_col + row*M + col];

        __syncthreads();
        #pragma unroll
        for (long d = 0; d < BLOCK_SIZE; ++d)
            temp += As[row][d]*Bs[d][col];
        
        __syncthreads();
    }

    if ((BLOCK_SIZE*block_col + col) < M && (BLOCK_SIZE*block_row + row) < N)
        c[M*BLOCK_SIZE*block_row + BLOCK_SIZE*block_col + row*M + col] = temp;
}


// b should be tranposed
template <typename scalar_t> 
__global__ void tiled_mm_kernel_with_transpose_2(
    const scalar_t* __restrict__ a, 
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    long N, long M, long D) {
    __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    long block_row = blockIdx.x;
    long block_col = blockIdx.y;

    scalar_t temp = 0.0f;
    long row = threadIdx.x;
    long col = threadIdx.y;

    #pragma unroll
    for (long k = 0; k <= (D / BLOCK_SIZE); ++k) {
        // load 
        if ((BLOCK_SIZE*k + col) >= D)
            As[row][col] = 0.0f;
        else 
            As[row][col] = a[D*BLOCK_SIZE*block_row + BLOCK_SIZE*k + row*D + col];
        
        if ((BLOCK_SIZE*k + col) >= D)
            Bs[col][row] = 0.0f;
        else 
            Bs[col][row] = b[D*BLOCK_SIZE*block_row + BLOCK_SIZE*k + row*D + col];  

        __syncthreads();
        #pragma unroll
        for (long d = 0; d < BLOCK_SIZE; ++d)
            temp += As[row][d]*Bs[d][col];
        
        __syncthreads();
    }

    if ((BLOCK_SIZE*block_col + col) < M && (BLOCK_SIZE*block_row + row) < N)
        c[M*BLOCK_SIZE*block_row + BLOCK_SIZE*block_col + row*M + col] = temp;
}



}


torch::Tensor segment_mm_cuda_forward(
    const torch::Tensor& mat_A, 
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A,
    const torch::Tensor& segment_id_B) {
    const long N = mat_A.size(0);
    const long M = mat_B.size(0);
    const long D = mat_A.size(1);

    cudaSetDevice(mat_A.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    auto count_A = segment_id_A.bincount().cpu();
    auto count_B = segment_id_B.bincount().cpu();
    auto accessor_A = count_A.accessor<long, 1>();
    auto accessor_B = count_B.accessor<long, 1>();

    // allocate C
    long sum = 0;
    const long size = count_A.size(0);
    for (long i = 0; i < size; ++i) {
        auto N_i = accessor_A[i];
        auto M_i = accessor_B[i];
        sum += N_i * M_i;
    }
    auto C = torch::zeros({sum}, mat_A.options());

    // loop k times, launch k kernels 
    long start_A = 0, start_B = 0, start_C = 0;    
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    for (long i = 0; i < size; ++i) {
        auto N_i = accessor_A[i];
        auto M_i = accessor_B[i];

        auto A_i = mat_A.narrow(0, start_A, N_i); // [N_i, D]
        auto B_i = mat_B.narrow(0, start_B, M_i); // [M_i, D] 
        
        dim3 dim_grid((N_i + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (M_i + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // async dispatch
        AT_DISPATCH_FLOATING_TYPES(mat_A.type(), "segment_mm_cuda_forward", ([&]{
            tiled_mm_kernel_with_transpose_2<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
               A_i.data<scalar_t>(),
               B_i.data<scalar_t>(),
               C.narrow(0, start_C, N_i*M_i).data<scalar_t>(),
               N_i, M_i, D);
        }));

        start_A += N_i;
        start_B += M_i;
        start_C += N_i * M_i;
    }

    return C;
}

std::vector<torch::Tensor> segment_mm_cuda_backward(
    const torch::Tensor& grad_c,
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A, 
    const torch::Tensor& segment_id_B) {
    const long N = mat_A.size(0);
    const long M = mat_B.size(0);
    const long D = mat_A.size(1);

    auto dA = torch::zeros({N, D}, grad_c.options());
    auto dB = torch::zeros({M, D}, grad_c.options());
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    cudaSetDevice(grad_c.get_device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto count_A = segment_id_A.bincount().cpu();
    auto count_B = segment_id_B.bincount().cpu();
    auto accessor_A = count_A.accessor<long, 1>();
    auto accessor_B = count_B.accessor<long, 1>();

    // calculate dA & dB
    long size = count_A.size(0);
    long start_A = 0, start_B = 0;
    long start_dA = 0, start_dB = 0, start_dC = 0;
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    for (long i = 0; i < size; ++i) {
        auto N_i = accessor_A[i];
        auto M_i = accessor_B[i];

        // dA_i = dC_i @ B_i
        auto dA_i = dA.narrow(0, start_dA, N_i);
        auto B_i = mat_B.narrow(0, start_B, M_i);
        auto dC_i = grad_c.narrow(0, start_dC, N_i * M_i);

        
        // [N_i, M_i] @ [M_i, D] -> [N_i, D]
        dim3 dim_grid_0((N_i + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (D + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // async dispatch
        AT_DISPATCH_FLOATING_TYPES(mat_A.type(), "segment_mm_cuda_backward_0", ([&]{
            tiled_mm_kernel<scalar_t><<<dim_grid_0, dim_block, 0, stream>>>(
                dC_i.data<scalar_t>(),
                B_i.data<scalar_t>(),
                dA_i.data<scalar_t>(),
                N_i, D, M_i
            );
        }));

        start_dA += N_i;
        start_B += M_i;

        // dB_i = dC_i^T @ A_i
        auto A_i = mat_A.narrow(0, start_A, N_i);
        auto dB_i = dB.narrow(0, start_dB, M_i);

        // [N_i, M_i]^T @ [N_i, D] -> [M_i, D]
        dim3 dim_grid_1((M_i + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                        (D + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // async dispatch
        AT_DISPATCH_FLOATING_TYPES(mat_A.type(), "segment_mm_cuda_backward_1", ([&]{
            tiled_mm_kernel_with_transpose_1<scalar_t><<<dim_grid_1, dim_block, 0, stream>>>(
                dC_i.data<scalar_t>(),
                A_i.data<scalar_t>(),
                dB_i.data<scalar_t>(),
                M_i, D, N_i
            );
        }));

        start_A += N_i;
        start_dB += M_i;

        start_dC += N_i * M_i;
    }

    return {dA, dB};
}
