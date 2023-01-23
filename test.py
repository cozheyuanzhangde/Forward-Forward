import torch
a = torch.arange(12, dtype= torch.float).reshape(3,4)
print(a.norm(2,1))
print(a.mean(1))
print(a.mean(1, True))