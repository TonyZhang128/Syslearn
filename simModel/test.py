import torch
import torch.nn as nn
from zyn.utils import cross_entropy
output = torch.tensor([[1,0.2,0.3],[1,0.2,0.6]])
label = torch.tensor([0,0])

loss = nn.CrossEntropyLoss()
loss_1 = loss(output, label)
loss_2 = cross_entropy(output, label)

print()