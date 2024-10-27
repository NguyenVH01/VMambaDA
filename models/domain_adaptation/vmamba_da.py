from re import X
from typing import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from models.domain_adaptation.grad_reverse import grad_reverse
from models.vmamba import Backbone_VSSM

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Feature(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(Feature, self).__init__()
        self.pretrained = pretrained
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3),
                                     pretrained=self.pretrained,
                                     **kwargs)

        self.num_features = self.encoder.num_features
        self.feature_extract = nn.Sequential(OrderedDict(
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
        ))

    def forward(self, x):
      x = self.encoder(x) # B x C x H x W
      x = self.feature_extract(x[-1])
      return x


class Predictor(nn.Module):
    def __init__(self, num_features, num_classes=3, dropout_prob=0.5):
        super(Predictor, self).__init__()

        self.feature_size = num_features
        

        self.class_classifier = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.BatchNorm1d(self.feature_size),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.BatchNorm1d(self.feature_size // 2),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_size // 2, num_classes)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)

        x = self.class_classifier(x)
        return x