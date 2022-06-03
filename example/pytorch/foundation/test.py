import torch

mat = torch.rand((3,5))

maxIndex = torch.argmax(mat,dim=1)

