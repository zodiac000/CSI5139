import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from pdb import set_trace

class VGG16(nn.Module):
    def __init__(self, clf_pth=''):
        super().__init__()
        self.entry = nn.Conv2d(1, 3, 3, padding=1)
        self.features = vgg16(pretrained=True).features
        self.clf_pth = clf_pth
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 5),
            nn.LogSoftmax(dim=1),
        )
        if not clf_pth:
            print("vgg features are locked.")
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            print("vgg features are not locked.")
            print("clf_pth is loaded for vgg ")
            self.load_state_dict(torch.load(clf_pth))


            for param in self.features.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.entry(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
