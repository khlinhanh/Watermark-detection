import torch.nn as nn
from torchvision import models

class WatermarkNet(nn.Module):
    def __init__(self, num_classes=4, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone=='resnet18':
            self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.cls_head = nn.Sequential(nn.Linear(feat_dim,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,num_classes))
        self.reg_head = nn.Sequential(nn.Linear(feat_dim,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,1))
        self.bbox_head = nn.Sequential(nn.Linear(feat_dim,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,4))

    def forward(self,x):
        f = self.base(x)
        cls = self.cls_head(f)
        reg = self.reg_head(f)
        bbox = self.bbox_head(f)
        return cls, reg, bbox
