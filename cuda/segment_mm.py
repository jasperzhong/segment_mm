import torch 
import torch.nn as nn 
from torch.autograd import Function 

import segment_mm

class SegmentMMFunction(Function):
    @staticmethod
    def forward(ctx, mat_A, mat_B, segment_id_A, segment_id_B):
        return segment_mm.forward(mat_A, mat_B, segment_id_A, segment_id_B)
    
    @staticmethod
    def backward(ctx, grad_c, mat_A, mat_B, segment_id_A, segment_id_B):
        return segment_mm.backward(grad_c, mat_A, mat_B, segment_id_A, segment_id_B)

class SegmentMM(nn.Module):
    def __init__(self, ):
        super(SegmentMM, self).__init__()

    def forward(self, mat_A, mat_B, segment_id_A, segment_id_B):
        return SegmentMMFunction.apply(mat_A, mat_B, segment_id_A, segment_id_B)