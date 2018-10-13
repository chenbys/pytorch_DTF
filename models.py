import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeformableConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None):
        super(DeformableConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size * stride, bias=bias)

    def forward(self, feature, offset):
        '''

        :param feature: [32,128,28,28]
        :param offset: [32,18,28,28]
        :return:
        '''

        B, C, H, W = feature.shape
        k = self.kernel_size

        source_idx_x = range(0, W)
        source_idx_y = range(0, H)

        fgrid = torch.randn([B, H * k, W * k, 2], requires_grad=False).type_as(feature).cuda()
        for x in source_idx_x:
            for y in source_idx_y:
                anchor_ = ([x, y] - np.array([W - 1, H - 1]) / 2.) / (np.array([W - 1, H - 1]) / 2.)
                anchor = torch.tensor(anchor_, requires_grad=False).type_as(feature).cuda()

                kernel_offset_ = offset[:, :, x, y]
                kernel_offset = kernel_offset_.view((B, k, k, 2))

                target = kernel_offset + anchor

                # size:[32,3,3,2]
                fgrid[:, k * y:k * y + k, k * x:k * x + k, :] = target

        deformed_feature = F.grid_sample(feature, fgrid, mode='bilinear')

        def compare(R=range(0, 5)):
            '''
                debug function

            :param R: the list of img idxs you want to show
            :return:
            '''

            f = plt.figure(figsize=(4, 10))
            N = len(R)
            for i in R:
                source = feature[i, 0, :, :].detach().cpu().numpy()
                target = deformed_feature[i, 0, :, :].detach().cpu().numpy()
                x1 = plt.subplot(N, 2, 2 * i + 1)
                x2 = plt.subplot(N, 2, 2 * i + 2)
                x1.imshow(source)
                x2.imshow(target)
            f.show()

        return self.conv_kernel(deformed_feature)


class DeformableConvNetwork(nn.Module):
    def __init__(self):
        super(DeformableConvNetwork, self).__init__()
        self.offset_conv = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.dcn = DeformableConv(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.dcn(x, offset)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformableConv(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        offsets = self.offsets(x)
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=28, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)
