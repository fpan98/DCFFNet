import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np


class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base, self).__init__()
        with torch.no_grad():
            features = list(vgg16(pretrained=True).features)[:23]
            self.features = nn.ModuleList(features)

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)
        return results


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):  # INF原文中是8，为了减少参数量，我改成了16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True, out=None)[0]

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class VggBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = conv2d_bn(in_channels, middle_channels)
        self.conv2 = conv2d_bn(middle_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class AttnIF(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = conv3x3(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        return out


class IFN_NestedUnet(nn.Module):
    def __init__(self, vgg16=vgg16_base()):
        super().__init__()
        self.base = vgg16
        nb_filter = [32, 64, 128, 256, 512]
        fe_filter = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)

        self.up1_0 = up(nb_filter[1])
        self.up2_0 = up(nb_filter[2])
        self.up1_1 = up(nb_filter[1])
        self.up3_0 = up(nb_filter[3])
        self.up2_1 = up(nb_filter[2])
        self.up1_2 = up(nb_filter[1])
        self.up4_0 = up(nb_filter[4])
        self.up3_1 = up(nb_filter[3])
        self.up2_2 = up(nb_filter[2])
        self.up1_3 = up(nb_filter[1])

        self.conv0_0 = VggBlock(6, nb_filter[0], nb_filter[0])
        self.conv1_0 = VggBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VggBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VggBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VggBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VggBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VggBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VggBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VggBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VggBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VggBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VggBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VggBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VggBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VggBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])

        self.IF1_0 = AttnIF(in_channels=nb_filter[1] + fe_filter[0] * 2, mid_channels=nb_filter[1]*2, out_channels=nb_filter[1])

        self.IF1_1 = AttnIF(in_channels=nb_filter[2] + fe_filter[1] * 2, mid_channels=nb_filter[2]*2, out_channels=nb_filter[2])
        self.IF0_2 = AttnIF(in_channels=nb_filter[1] + fe_filter[0] * 2, mid_channels=nb_filter[1]*2, out_channels=nb_filter[1])

        self.IF2_1 = AttnIF(in_channels=nb_filter[3] + fe_filter[2] * 2, mid_channels=nb_filter[3]*2, out_channels=nb_filter[3])
        self.IF1_2 = AttnIF(in_channels=nb_filter[2] + fe_filter[1] * 2, mid_channels=nb_filter[2]*2, out_channels=nb_filter[2])
        self.IF0_3 = AttnIF(in_channels=nb_filter[1] + fe_filter[0] * 2, mid_channels=nb_filter[1]*2, out_channels=nb_filter[1])

        self.IF3_1 = AttnIF(in_channels=nb_filter[4] + fe_filter[3] * 2, mid_channels=nb_filter[4]*2, out_channels=nb_filter[4])
        self.IF2_2 = AttnIF(in_channels=nb_filter[3] + fe_filter[2] * 2, mid_channels=nb_filter[3]*2, out_channels=nb_filter[3])
        self.IF1_3 = AttnIF(in_channels=nb_filter[2] + fe_filter[1] * 2, mid_channels=nb_filter[2]*2, out_channels=nb_filter[2])
        self.IF0_4 = AttnIF(in_channels=nb_filter[1] + fe_filter[0] * 2, mid_channels=nb_filter[1]*2, out_channels=nb_filter[1])

        self.final1 = nn.Conv2d(nb_filter[0], 1, 1)
        self.final2 = nn.Conv2d(nb_filter[0], 1, 1)
        self.final3 = nn.Conv2d(nb_filter[0], 1, 1)
        self.final4 = nn.Conv2d(nb_filter[0], 1, 1)

        self.ca = ChannelAttention(nb_filter[0] * 4, ratio=16)
        self.sa = SpatialAttention()
        self.conv_1x1 = nn.Conv2d(nb_filter[0] * 4, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1_input, t2_input):
        t1_list = self.base(t1_input)
        t2_list = self.base(t2_input)
        t1_f_l3, t1_f_l8, t1_f_l15, t1_f_l22 = t1_list[0], t1_list[1], t1_list[2], t1_list[3]
        t2_f_l3, t2_f_l8, t2_f_l15, t2_f_l22 = t2_list[0], t2_list[1], t2_list[2], t2_list[3]

        x0_0 = self.conv0_0(torch.cat((t1_input, t2_input), dim=1))
        x1_0 = self.conv1_0(self.pool(x0_0))
        IF1_0 = self.up1_0(x1_0)
        IF1_0 = self.IF1_0(torch.cat((IF1_0, t1_f_l3, t2_f_l3), dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, IF1_0], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        IF1_1 = self.up2_0(x2_0)
        IF1_1 = self.IF1_1(torch.cat((IF1_1, t1_f_l8, t2_f_l8), dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, IF1_1], 1))
        IF0_2 = self.up1_1(x1_1)
        IF0_2 = self.IF0_2(torch.cat((IF0_2, t1_f_l3, t2_f_l3), dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, IF0_2], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        IF2_1 = self.up3_0(x3_0)
        IF2_1 = self.IF2_1(torch.cat((IF2_1, t1_f_l15, t2_f_l15), dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, IF2_1], 1))
        IF1_2 = self.up2_1(x2_1)
        IF1_2 = self.IF1_2(torch.cat((IF1_2, t1_f_l8, t2_f_l8), dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, IF1_2], 1))
        IF0_3 = self.up1_2(x1_2)
        IF0_3 = self.IF0_3(torch.cat((IF0_3, t1_f_l3, t2_f_l3), dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, IF0_3], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        IF3_1 = self.up4_0(x4_0)
        IF3_1 = self.IF3_1(torch.cat((IF3_1, t1_f_l22, t2_f_l22), dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, IF3_1], 1))
        IF2_2 = self.up3_1(x3_1)
        IF2_2 = self.IF2_1(torch.cat((IF2_2, t1_f_l15, t2_f_l15), dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, IF2_2], 1))
        IF1_3 = self.up2_2(x2_2)
        IF1_3 = self.IF1_2(torch.cat((IF1_3, t1_f_l8, t2_f_l8), dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, IF1_3], 1))
        IF0_4 = self.up1_3(x1_3)
        IF0_4 = self.IF0_3(torch.cat((IF0_4, t1_f_l3, t2_f_l3), dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, IF0_4], 1))

        # 四个中间输出做深监督
        output1 = self.sigmoid(self.final1(x0_1))
        output2 = self.sigmoid(self.final2(x0_2))
        output3 = self.sigmoid(self.final3(x0_3))
        output4 = self.sigmoid(self.final4(x0_4))

        out = torch.cat([x0_1, x0_2, x0_3, x0_4], 1)
        out = self.ca(out)*out
        out = self.sa(out)*out
        out = self.sigmoid(self.conv_1x1(out))
        return [output1, output2, output3, output4, out]
