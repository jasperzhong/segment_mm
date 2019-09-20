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
    print(torch.allclose(c1,c2))

def test_backward():
    a = torch.randn(3, 3, requires_grad=True).cuda()
    b = torch.randn(4, 3, requires_grad=True).cuda()

    segment_id_a = torch.tensor([0]*1 + [1]*1 + [2]*1).cuda()
    segment_id_b = torch.tensor([0]*2 + [1]*1 + [2]*1).cuda()

    model = SegmentMM()

    c = model(a, b, segment_id_a, segment_id_b)
    x = torch.sum(c)
    a.retain_grad()
    b.retain_grad()
    x.backward()

    grad_A = a.grad.clone()
    grad_B = b.grad.clone()

    a.grad.zero_()
    b.grad.zero_()
    c2 = naive_segment_mm(a, b, segment_id_a, segment_id_b)
    x2 = torch.sum(c2)
    x2.backward()
    grad_A_true = a.grad
    grad_B_true = b.grad

    print(torch.allclose(grad_A, grad_A_true))
    print(torch.allclose(grad_B, grad_B_true))


if __name__=='__main__':
    test_backward()