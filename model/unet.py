# -*- coding: utf-8 -*-

from torch import nn
from torchtoolbox.nn.init import KaimingInitializer
import torch


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block2_1 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(48, 48, 3, 2, 1, output_padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2_1(pool1)
        pool3 = self._block2_2(pool2)
        pool4 = self._block2_3(pool3)
        pool5 = self._block2_4(pool4)

        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        out = self._block6(concat1)
        return out


if __name__ == '__main__':
    model = UNet()
    initializer = KaimingInitializer(model)
    a = torch.rand(1, 3, 256, 256)
    out = model(a)
    print(out)