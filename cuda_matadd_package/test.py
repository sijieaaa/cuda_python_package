import torch
import cuda_torch_extension


A = torch.rand(3, 3).cuda()
B = torch.rand(3, 3).cuda()
C = torch.zeros_like(A).cuda()
cuda_torch_extension.add(A, B, C)

print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Matrix C (Result):")
print(C)
