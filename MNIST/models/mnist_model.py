import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class MNISTModel(Module):
    def __init__(self, opt):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # flatten 1 mean get an 2 dim result
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)

        output = self.dropout2(x)

        
        return output

