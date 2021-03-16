import timm
import torch.nn as nn


class CustomResnet(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=False, target_size=0):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, target_size)

    def forward(self, x):
        x = self.model(x)
        return x
