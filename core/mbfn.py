from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion, relu=False):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.ReLU(inplace=True) if relu else nn.PReLU(inp * expansion),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.ReLU(inplace=True) if relu else nn.PReLU(inp * expansion),

            #pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False, relu=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.ReLU(inplace=True) if relu else nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting, feat_dim=512, relu=False):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1, relu=relu)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True, relu=relu)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting, relu)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0, relu=relu)
        
        self.linear7 = ConvBlock(512, 512, (14, 14), 1, 0, dw=True, linear=True, relu=relu)

        self.linear1 = ConvBlock(512, feat_dim, 1, 1, 0, linear=True, relu=relu)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.flatten(start_dim=1)
        #x = x.view(x.size(0), -1)

        return x
    
    def visualization(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        return x

if __name__ == "__main__":
    from model_complexity import compute_model_complexity

    net = MobileFacenet(feat_dim=512, relu=False)
    net.eval()
    input_size = 224
    num_params, flops = compute_model_complexity(
        net, (1, 3, input_size, input_size)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))
