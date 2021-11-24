import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class MNISTModel(Module):
    def __init__(self, opt):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        # self.fc2 = nn.Linear(2048, 1024)
        # self.fc3 = nn.Linear(1024,512)
        # self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        # flatten 1 mean get an 2 dim result
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)

        # x = self.fc2(x)
        # x = self.dropout1(x)
        # x = F.relu(x)

        # x = self.fc3(x)
        # # x = self.dropout1(x)
        # x = F.relu(x)

        # x = self.fc4(x)
        # x = self.dropout1(x)
        # x = F.relu(x)

        x = self.fc5(x)
        # x = self.dropout2(x)
        # x = F.relu(x)

        output = x

        
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        output = x
        return output