import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, vit_h_14, ViT_H_14_Weights, vit_l_16, ViT_L_16_Weights, vit_b_16, ViT_B_16_Weights

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.fc = nn.Linear(1000, nclasses)
    
    def forward(self, x):
        x = F.relu(self.pretrained(x))
        return self.fc(x)

class ViTh14(nn.Module):
    def __init__(self):
        super(ViTh14, self).__init__()
        self.pretrained = vit_h_14(ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        self.fc = nn.Linear(1000, nclasses)

    def forward(self, x):
        x = F.relu(self.pretrained(x))
        return self.fc(x)
    
class ViTl16(nn.Module):
    def __init__(self):
        super(ViTl16, self).__init__()
        self.pretrained = vit_l_16(ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.fc = nn.Linear(1000, nclasses)

    def forward(self, x):
        x = F.relu(self.pretrained(x))
        return self.fc(x)
    
class ViTb16(nn.Module):
    def __init__(self):
        super(ViTb16, self).__init__()
        self.pretrained = vit_b_16(ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.fc = nn.Linear(1000, nclasses)

    def forward(self, x):
        x = F.relu(self.pretrained(x))
        return self.fc(x)