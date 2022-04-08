import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from groupcnn import ConvP4, ConvZ2P4, MaxRotationPoolP4, MaxSpatialPoolP4


class MultiClassImageClassifier(nn.Module):
    def __init__(self):
        super(MultiClassImageClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 3)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = F.softmax(x, dim=1)
        return x


class EquiNet(torch.nn.Module):
    def __init__(self):
        super(EquiNet, self).__init__()
        self.conv1 = ConvZ2P4(1, 8, 5)
        self.pool1 = MaxSpatialPoolP4(2)
        self.conv2 = ConvP4(8, 32, 3)
        self.pool2 = MaxSpatialPoolP4(2)
        self.conv3 = ConvP4(32, 64, 3)
        self.pool3 = MaxSpatialPoolP4(2)
        self.conv4 = ConvP4(64, 32, 3)
        self.pool4 = MaxRotationPoolP4()
        self.pool5 = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(32, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(self.pool1(x))
        x = self.conv2(x)
        x = torch.nn.functional.relu(self.pool2(x))
        x = self.conv3(x)
        x = torch.nn.functional.relu(self.pool3(x))
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output
