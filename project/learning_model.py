import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class FLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv1d(1, 16, 2)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 16, 2)
        self.maxpool2 = nn.MaxPool1d(2)
        # self.Flatten = nn.Flatten()
        self.l1 = nn.Linear(240, 512)
        # self.dropout = nn.Dropout(0.5)
        # self.dropout = nn.Dropout(0.2)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 14)
        # self.re = torch.reshape()

    # normal: (batch_size, 3d-tensor)
    # now: (batch_size, 1d-tensor) (320, 63)
    def forward(self, x):
        # x = torch.reshape(x, (2, 1, 63))
        x = x.unsqueeze(1)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print("卷积后..............")
        # print(x.shape)
        x = self.maxpool1(x)
        # print("池化后..............")
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size()[0], -1)
        # print("flatten.................")
        # print(x.shape)

        # x = self.l1(x)
        # x = F.dropout(x,p=0.2)
        # x = F.relu(x)
        # x = self.l3(x)
        # x = F.dropout(x,p=0.2)
        # x = F.relu(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        output = F.log_softmax(self.l5(x), dim=1)
        # output: (batchsize, classifyResult)
        return output
