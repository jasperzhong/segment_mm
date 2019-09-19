import torch 
import torch.nn as nn

import segment_mm

from cuda.segment_mm import SegmentMM
from python.segment_mm import bmm_segment_mm, naive_segment_mm

def test_forward():
    a = torch.randn(12, 200).cuda()
    b = torch.randn(15, 200).cuda()

    segment_id_a = torch.tensor([0]*4 + [1]*4 + [2]*4).cuda()

    segment_id_b = torch.tensor([0]*5 + [1]*5 + [2]*5).cuda()

    c1 = segment_mm.forward(a, b, segment_id_a, segment_id_b)
    c2 = bmm_segment_mm(a, b, segment_id_a, segment_id_b)
    c3 = naive_segment_mm(a, b, segment_id_a, segment_id_b)
    print(torch.max(torch.abs(c1-c2)).cpu().item())

def test_backward():
    model = SegmentMM()

    a = torch.randn(12, 200, requires_grad=True).cuda()
    b = torch.randn(15, 200, requires_grad=True).cuda()
    segment_id_a = torch.tensor([0]*4 + [1]*4 + [2]*4).cuda()
    segment_id_b = torch.tensor([0]*5 + [1]*5 + [2]*5).cuda()

    c = model(a, b, segment_id_a, segment_id_b)
    x = sum(c)
    x.backward()
    
    

if __name__=="__main__":
    test_backward()