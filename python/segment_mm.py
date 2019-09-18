import torch 

# too young, too simple, sometimes naive
def naive_segment_mm(mat_A, mat_B, segment_id_A, segment_id_B):
    count_A = segment_id_A.bincount()
    count_B = segment_id_B.bincount()
    
    C_list = []
    start_A, start_B, start_C = 0, 0, 0
    for N_i, M_i in zip(count_A, count_B):
        A_i = mat_A.narrow(0, start_A, N_i)
        B_i = mat_B.narrow(0, start_B, M_i).transpose(0, 1)
        C_i = (A_i @ B_i).flatten()
        C_list.append(C_i)
        start_A += N_i 
        start_B += M_i

    return torch.cat(C_list, 0)        

# padding and bmm
def bmm_segment_mm(mat_A, mat_B, segment_id_A, segment_id_B):
    device = mat_A.get_device()

    count_A = segment_id_A.bincount()
    count_B = segment_id_B.bincount()
    
    k = len(count_A)
    B = max(torch.max(count_A).item(), torch.max(count_B).item())
    D = mat_A.size(1)

    # padding to [B, D], stack to [k, B, D]
    As = []
    start_A = 0
    for N_i in count_A:
        A_i = torch.cat((mat_A.narrow(0, start_A, N_i), torch.zeros(B-N_i, D).to(device)), 0)
        As.append(A_i)
        start_A += N_i
    A_padded = torch.stack(As, 0)

    Bs = []
    start_B = 0
    for M_i in count_B:
        B_i = torch.cat((mat_B.narrow(0, start_B, M_i), torch.zeros(B-M_i, D).to(device)), 0)
        Bs.append(B_i)
        start_B += M_i
    B_padded = torch.stack(Bs, 0)

    # [k, B, B]
    C_padded = torch.bmm(A_padded, B_padded.transpose(1,2))

    Cs = []
    start_C = 0
    for i, (N_i, M_i) in enumerate(zip(count_A, count_B)):
        Cs.append(C_padded[i][:N_i][:M_i].flatten())
    
    return torch.cat(Cs, 0)


if __name__=="__main__":
    a = torch.randn(12, 5)
    b = torch.randn(15, 5)
    segment_id_a = torch.tensor([0]*4 + [1]*4 + [2]*4)
    segment_id_b = torch.tensor([0]*5 + [1]*5 + [2]*5)
    c1 = naive_segment_mm(a, b, segment_id_a, segment_id_b)
    c2 = bmm_segment_mm(a, b, segment_id_a, segment_id_b)

    print("max error:", torch.max(c1 - c2).item())

    