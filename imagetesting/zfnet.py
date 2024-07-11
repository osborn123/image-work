# import torch.nn as nn
# import torch.nn.functional as F

# class ZFNet(nn.Module):
#     """ A implementation of the ZFNet architecture from the paper 'Visualizing and 
#         Understanding Convolutional Networks' by Zeiler and Fergus"""
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3, padding_mode='reflect') 
#         self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, padding_mode='reflect')
#         self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
#         self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
#         self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
#         self.fc6 = nn.Linear(9216,4096)
#         self.fc7 = nn.Linear(4096,4096)
#         self.fc8 = nn.Linear(4096,1000)
#         self.pool1 = nn.MaxPool2d(3,stride=2)
#         self.pool2 = nn.MaxPool2d(3,stride=2)
#         self.drop = nn.Dropout(0.5)
#         self.drop = nn.Dropout(0.5)
#         self.lrn = nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0)

#     def forward(self, x):
#         x = self.lrn(self.pool1(F.relu(self.conv1(x))))
#         x = self.lrn(self.pool2(F.relu(self.conv2(x))))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool2(F.relu(self.conv5(x)))
#         x = x.view(-1,9216)
#         x = F.relu(self.drop(self.fc6(x)))
#         x = F.relu(self.drop(self.fc7(x)))
#         x = self.fc8(x)
#         return x

# zfnet.py

import torch
import torch.nn as nn

class ZFNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ZFNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
