import numpy as np
import torch
a = np.array([1,2,3])
d = np.array([1,2,4])
b = np.array([[1], [2], [3]])
c = b ** b
print(a**a)
print(b**b)
print(b[2:, :])
print(c.sum())
print(b[0:-1, :])
c = torch.tensor([[1., 2., 3.],
                   [-1, 1, 4]])
print(torch.linalg.norm(c, dim=1))
e = np.array([[1,2,3],[2,3,4]])
print(a * d)
print(a * e)
exponential = None
device='cpu'
N = 5
# This binary mask zeros out terms where k=i.
mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
print(mask)
# We apply the binary mask.
exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
print(exponential)