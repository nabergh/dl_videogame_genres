import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GenreClassifier(torch.nn.Module):
    def __init__(self, dtype, num_genres):
        super(GenreClassifier, self).__init__()

        self.dtype = dtype
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout()
        self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        # self.bn6 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 128
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 64
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 32
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # 16
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        # 8
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        #4
        
        self.lin1 = nn.Linear(8192, 1024)
        self.lin2 = nn.Linear(1024, 256)
        self.lin3 = nn.Linear(256, num_genres)
        
        self.type(dtype)
        self.train()

    def forward(self, images):
        x = self.pool(F.leaky_relu(self.conv1(images)))
        x = self.bn1(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        # x = self.bn2(x)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(F.leaky_relu(self.conv4(x)))
        # x = self.bn4(x)
        x = self.pool(F.leaky_relu(self.conv5(x)))
        x = self.bn5(x)
        x = self.pool(self.dropout(F.leaky_relu(self.conv6(x))))
        # x = self.bn6(x)

        x = x.view(images.size(0), -1)

        x = self.dropout(F.leaky_relu(self.lin1(x)))
        x = self.dropout(F.leaky_relu(self.lin2(x)))
        return F.sigmoid(self.lin3(x))
