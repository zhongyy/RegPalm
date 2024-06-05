from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter

from .stn import STN
from .mbfn import MobileFacenet
import numpy as np

class STN_MBFN(nn.Module):
    def __init__(self, feat_dim=512, aug_size=250):
        super(STN_MBFN, self).__init__()
        self.backbone = MobileFacenet(feat_dim=feat_dim)

        self.stn = STN(input_size=aug_size) 
    
    def forward(self, x):
        x = self.stn(x)
        x = self.backbone(x)
        return x
