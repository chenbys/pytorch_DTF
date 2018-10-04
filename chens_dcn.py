import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt

from torch import nn
from torch.autograd import Variable


class DCN(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None):
        super(DCN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size * stride, bias=bias)

    def forward(self, feature, offset):
        '''

        :param x: [32,128,28,28]
        :param offset: [32,18,28,28]
        :return:
        '''

        ESP = 1e-9
        B, C, H, W = feature.shape
        k = self.kernel_size

        source_idx_x = range(0, W)
        source_idx_y = range(0, H)

        fgrid = torch.randn([B, H * k, W * k, 2], requires_grad=True).type_as(feature).cuda()
        for x in source_idx_x:
            for y in source_idx_y:
                anchor_ = ([x, y] - np.array([W - 1, H - 1]) / 2.) / (np.array([W - 1, H - 1]) / 2.)
                anchor = torch.tensor(anchor_, requires_grad=True).type_as(feature).cuda()

                kernel_offset_ = offset[:, :, x, y]
                kernel_offset = kernel_offset_.view((B, k, k, 2))

                target = kernel_offset + anchor

                # size:[32,3,3,2]
                fgrid[:, k * y:k * y + k, k * x:k * x + k, :] = target

        transformed_feature = F.grid_sample(feature, fgrid, mode='bilinear')
        return self.conv_kernel(transformed_feature)
