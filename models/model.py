import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import VGG16
from models.msrn import MSRN

class Model(nn.Module):
    def __init__(self, clf_pth=''):
        super().__init__()
        self.sr = MSRN(4)
        self.sr.load_state_dict(torch.load("msrn.pth"))
        self.classifier = VGG16(clf_pth)

        for param in self.sr.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.sr(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Model()
