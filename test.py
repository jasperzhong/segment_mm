import torch 
import segment_mm

from python.segment_mm import bmm_segment_mm, naive_segment_mm

a = torch.randn(12, 5).cuda()
b = torch.randn(15, 5).cuda()

segment_id_a = torch.tensor([
    0,0,0,0,1,1,1,1,2,2,2,2
]).cuda()

segment_id_b = torch.tensor([
    0,0,0,0,0,1,1,1,1,1,2,2,2,2,2
]).cuda()

c1 = segment_mm.forward(a, b, segment_id_a, segment_id_b)
c2 = bmm_segment_mm(a, b, segment_id_a, segment_id_b)
c3 = naive_segment_mm(a, b, segment_id_a, segment_id_b)
print(c1)
print(c2)
print(c3)
print(torch.max(c1-c2).cpu().item())