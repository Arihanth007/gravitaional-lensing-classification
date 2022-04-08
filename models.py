import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models


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
