import torch 
import segment_mm

a = torch.randn(12, 5).cuda()
b = torch.randn(15, 5).cuda()

segment_id_a = torch.tensor([
    0,0,0,0,1,1,1,1,2,2,2,2
]).cuda()

segment_id_b = torch.tensor([
    0,0,0,0,0,1,1,1,1,1,2,2,2,2,2
]).cuda()

print(a)
print(b)
print(segment_id_a.type())
print(segment_mm.forward(a, b, segment_id_a, segment_id_b))