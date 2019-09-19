# segment_mm

This is an operator written in CUDA for PyTorch.

Compute the matrix multiplication between each pair of segments of two matrices, flatten the results to vectors, and cat them to one big vector.

```python
def segment_mm(
    mat_A, 
    mat_B, 
    segment_id_A, 
    segment_id_B):
    """
    Params:
    ------
        mat_A: float tensor, shape (N, D)
        mat_B: float tensor, shape (M, D)
        segment_id_A: long tensor, shape (N,), sorted list 
        segemtn_id_B: long tensor, shape (M,), sorted list
    
    Returns
    ------
        c: float tensor, shape (N_1*M_1 + N_2*M_2 + ... + N_k*M_k,)
           k is the number of segments
    """
```

Read [tf.math.segment_sum](https://www.tensorflow.org/api_docs/python/tf/math/segment_sum) for an explanation of segments.

![](https://s2.ax1x.com/2019/09/19/nL81Ug.png)

**For example**
```python
>>>import torch 
>>>from cuda.segment_mm import SegmentMM
>>>A = torch.FloatTensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]]).cuda()
>>>B = torch.FloatTensor([[4,3,2,1],[8,7,6,5]]).cuda()
>>>segment_id_A = torch.tensor([0,1,1]).cuda()
>>>segment_id_B = torch.tensor([0,1]).cuda()
>>>model = SegmentMM()
>>>c = model(A, B, segment_id_A, segment_id_B)
>>>c
tensor([ 20., 164., 268.], device='cuda:0')
```

# Benchmark

|  | forward(ms) | backward(ms) |
| --- | --- | --- |
| naive|6.437 |6.734 |
| bmm |15.656 |7.306 |
|cuda |**0.694** |**1.431** |

# Known Bugs

- [ ] When D is large(e.g. > 300), an error will occur in backward function of cuda version `RuntimeError: CUDA error: an illegal memory access was encountered`

# Reference 

- [custom_op_benchmark](https://github.com/yzh119/custom_op_benchmark)