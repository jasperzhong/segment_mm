import argparse
import time
import torch
import numpy as np
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['naive', 'bmm', 'cuda'])
parser.add_argument('-n', type=int, default=512)
parser.add_argument('-m', type=int, default=512)
parser.add_argument('-d', type=int, default=100)
parser.add_argument('-k', type=int, default=32)
parser.add_argument('-r', '--runs', type=int, default=1000)
options = parser.parse_args()

if options.example == 'naive':
    from python.segment_mm import naive_segment_mm
    segment_mm = naive_segment_mm
elif options.example == 'bmm':
    from python.segment_mm import bmm_segment_mm
    segment_mm = bmm_segment_mm
else:
    from cuda.segment_mm import SegmentMM
    segment_mm = SegmentMM()

N = options.n 
M = options.m 
D = options.d 
K = options.k 

mat_A = torch.randn(N, D, requires_grad=True).cuda()
mat_B = torch.randn(M, D, requires_grad=True).cuda()

segment_id_a = torch.tensor(np.repeat(list(range(K)), N//K)).cuda()
segment_id_b = torch.tensor(np.repeat(list(range(K)), M//K)).cuda()

# force cuda init
c = segment_mm(mat_A, mat_B, segment_id_a, segment_id_b)

forward_time = 0
backward_time = 0
for _ in tqdm(range(options.runs)):
    if options.example == 'cuda':
        segment_mm.zero_grad()

    start = time.time()
    c = segment_mm(mat_A, mat_B, segment_id_a, segment_id_b)
    elapsed = time.time() - start 
    forward_time += elapsed

    x = torch.sum(c)
    start = time.time()
    x.backward()
    elapsed = time.time() - start
    backward_time += elapsed

forward_time *= 1000.0
backward_time *= 1000.0

print('forward: %.3f  | backward: %.3f' % (
    forward_time/options.runs, backward_time/options.runs
))