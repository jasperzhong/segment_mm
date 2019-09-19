import argparse
import time
import torch

from cuda.segment_mm import SegmentMM
from python.segment_mm import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=128)
parser.add_argument('-m', type=int, default=128)
parser.add_argument('-k', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=100)
options = parser.parse_args()


mat_A = torch.randn()


cuda_segment_mm = SegmentMM()

# force cuda init
 

forward_time = 0
backward_time = 0
for _ in range(options.runs):
    start = time.time()

    elapsed = time.time() - start 
    forward_time += elapsed


    start = time.time()

    elapsed = time.time() - start
    backward_time += elapsed
