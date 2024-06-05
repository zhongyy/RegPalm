from torch import nn
import torch
import torch.nn.functional as F
#from torch.autograd import Variable

#from IPython import embed
import math
from .mbfn import MobileFacenet
#from torch.nn import Parameter

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        bottleneck_setting = [
            # t, c , n ,s
            [2, 16, 2, 2],
            [4, 32, 1, 2],
            [2, 32, 2, 1],
            [4, 64, 1, 2],
            [2, 64, 2, 1]
        ]

        self.encoder = MobileFacenet(
            bottleneck_setting=bottleneck_setting,
            feat_dim=256
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.aug_size = 250

        self.bn1d_wo_affine = nn.BatchNorm1d(6, affine=False)
        self.bn1d_scale = nn.Parameter(torch.Tensor(6))
        self.bn1d_scale.data.copy_(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float))


    def _make_layer(self, block, setting, relu):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t, relu=relu))
                else:
                    layers.append(block(self.inplanes, c, 1, t, relu=relu))
                self.inplanes = c

        return nn.Sequential(*layers)

    def foward_theta(self, x):
        x = F.interpolate(x, size=[224, 224], mode="bilinear")
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.bn1d_wo_affine(x) 
        x *= torch.clip(self.bn1d_scale, 2.5, 100)
        x = (F.sigmoid(x) - 0.5) * 2 * 0.1
        x += torch.tensor([224.0 / self.aug_size, 0, 0, 0, 224.0 / self.aug_size, 0]).to(x.device).view(-1, 6)
        x = x.view(-1, 2, 3)
        return x

    def stn(self, x):
        theta = self.foward_theta(x)
        grid = F.affine_grid(theta, (x.size(0), x.size(1), 224, 224))
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        return x
        
