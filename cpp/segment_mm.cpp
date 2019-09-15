#include <torch/extension.h>

#include <vector>
#include <THC/THC.h>

/*
    A1 @ B1 = C1 | 
    ...          | -> C   -- flatten -> c  
    Ak @ Bk = Ck |

    padding to the largest size and then do bmm
*/ 
torch::Tensor segment_mm_forward(
    const torch::Tensor& mat_A,
    const torch::Tensor& mat_B,
    const torch::Tensor& segment_id_A,
    const torch::Tensor& segment_id_B) {

    // check dim
    if (mat_A.dim() != 2 || mat_B.dim() != 2) {
        AT_ERROR("matrices should be a 2-D tensor");
    }

    if (segment_id_A.dim() != 1 || segment_id_A.dim() != 1) {
        AT_ERROR("segment_ids should be a 1-D tensor");
    }

    if (mat_A.size(1) != mat_B.size(1)) {
        AT_ERROR("matrices should be the same size at dimension 1");
    }

    if (segment_id_A.size(0) != mat_A.size(0) || segment_id_B.size(0) != mat_B.size(0)) {
        AT_ERROR("segment_ids should be the same size as dimension 0 of input");
    }

    // bincount: https://pytorch.org/docs/stable/torch.html#torch.bincount
    auto count_A = segment_id_A.bincount();
    auto count_B = segment_id_A.bincount();

    if (count_A.size(0) != count_B.size(0)) {
        AT_ERROR("the number of segments should be the same");
    }
    
    // only needs to check if there is zero value
    if (!count_A.is_nonzero() || !count_B.is_nonzero()) {
        AT_ERROR("segment_ids should be contiguous non-negative ints");
    }

    // check done

    // begin calculation
    auto block_size = std::max(count_A.max().item().toInt(), count_B.max().item().toInt());
    auto col_size = mat_A.size(1);
    
    // return a 3-D tensor [k, B, D]
    auto padding = [=](const auto& accessor, const torch::Tensor& mat) {
        std::vector<torch::Tensor> targets;
        auto size = accessor.size(0);
        long start = 0L, stride;
        for (long i = 0L; i < size; ++i) {
            stride = accessor[i];
            // do padding 
            auto padded = torch::cat({
                mat.narrow(0, start, stride),  // narrow() shares the same underlying storage
                torch::zeros({block_size - stride, col_size})}, 0);
            targets.push_back(std::move(padded));
            start += stride;
        }

        return torch::stack(targets);
    };

    // [k, B, D] @ [k, B, D]^T  -> [k, B, B]
    auto lhs = padding(count_A.accessor<long, 1>(), mat_A);
    auto rhs = padding(count_B.accessor<long, 1>(), mat_B);
    auto out = torch::bmm(lhs, rhs.transpose_(1, 2));
    
    // flatten 
    auto k = count_A.size(0);
    std::vector<torch::Tensor> vectors; 
    for (long i = 0L; i < k; ++i) {
        auto N_i = count_A[i].item().toLong();
        auto M_i = count_B[i].item().toLong();
        // extract submat (N_i, M_i) 
        auto vector = out[i].narrow(0, 0, N_i).narrow(1, 0, M_i).flatten();
        vectors.push_back(std::move(vector));
    }

    return torch::cat(vectors, 0);  
}


/*
    dC1 @ B1 = dA1 |
    ...            | -> dA
    dCk @ Bk = dAk | 

    dC1^T @ A1 = dB1 |
    ...              | -> dB
    dCk^T @ Ak = dBk |

    padding to the largest size and then do bmm
*/ 
std::vector<torch::Tensor> segment_mm_backward(
    torch::Tensor grad_c,
    torch::Tensor mat_A,
    torch::Tensor mat_B,
    torch::Tensor segment_id_A,
    torch::Tensor segment_id_B) {
    auto count_A = segment_id_A.bincount();
    auto count_B = segment_id_B.bincount();
    auto block_size = std::max(count_A.max().item().toInt(), count_B.max().item().toInt());
    
    // transform vector grad_c to {dC1 ... dCk} and padding to B
    std::vector<torch::Tensor> targets;
    long start = 0L, stride;
    auto k = count_A.size(0);
    for (int i = 0; i < k; ++i) {
        auto N_i = count_A[i].item().toLong();
        auto M_i = count_B[i].item().toLong();
        stride = N_i * M_i;
        auto dC_i = grad_c.narrow(0, start, stride).view({N_i, M_i});

        // padding to (B, B)
        auto dC_i_padded_dim0 = torch::cat({
           dC_i,
           torch::zeros({block_size-N_i, M_i}) 
        }, 0);
        auto dC_i_padded = torch::cat({
            dC_i_padded_dim0,
            torch::zeros({block_size, block_size-M_i})
        }, 1);

        targets.push_back(std::move(dC_i_padded));
    }

    // [k, B, B]
    auto dC = torch::stack(targets);

    // padding mat_A and mat_B 
    auto col_size = mat_A.size(1);
    auto padding = [=](const auto& accessor, const torch::Tensor& mat) {
        std::vector<torch::Tensor> targets;
        auto size = accessor.size(0);
        long start = 0L, stride;
        for (long i = 0L; i < size; ++i) {
            stride = accessor[i];
            // do padding 
            auto padded = torch::cat({
                mat.narrow(0, start, stride),  // narrow() shares the same underlying storage
                torch::zeros({block_size - stride, col_size})}, 0);
            targets.push_back(std::move(padded));
            start += stride;
        }

        return torch::stack(targets);
    };

    // [k, B, D]
    auto A = padding(count_A.accessor<long, 1>(), mat_A);
    auto B = padding(count_B.accessor<long, 1>(), mat_B);

    // [k, B, B] @ [k, B, D] -> [k, B, D]
    auto dA = torch::bmm(dC, B);

    // [k, B, B]^T @ [k, B, D] -> [k, B, D]
    auto dB = torch::bmm(dC.transpose(1, 2), A);

    // extract (N_i, D) or (M_i, D)
    auto extract = [=](const torch::Tensor& grad, const auto& accessor) {
        std::vector<torch::Tensor> grad_segments;
        for (int i = 0;i < k; ++i) {
            grad_segments.push_back(grad[i].narrow(0, 0, accessor[i]));
        }
        return torch::cat(grad_segments, 0);
    };  
    
    return {extract(dA, count_A.accessor<long, 1>()),
            extract(dB, count_B.accessor<long, 1>())};
}


PYBIND11_MODULE(TORCH_EXTENSION, m) {
    m.def("forward", &segment_mm_forward, "segment_mm forward");
    m.def("backward", &segment_mm_backward, "segment_mm backward");
}