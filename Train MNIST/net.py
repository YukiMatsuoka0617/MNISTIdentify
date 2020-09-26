import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(3*3*256, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(-1, 3*3*256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)