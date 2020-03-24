# -*- coding: utf-8 -*-
__all__ = ['ResNoise', 'ResConnNoise']

from torch import nn
import torch


class Residual(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_c, hid_c, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hid_c, out_c, 1, 1)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers):
        super(ResidualStack, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(Residual(in_c, hid_c, out_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.block = nn.Sequential(
            # 256^2 x 3
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(True),
            # 128^2 x 16
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(True),
            # 64^2 x 32
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(True),
            # 32^2 x 64
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            # 16^2 x 128
            nn.Conv2d(128, 256, 3, 2, 1),
            # 8^2 x 256
            ResidualStack(256, 64, 256, 2),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.block = nn.Sequential(
            # 8^2 x 256
            ResidualStack(256, 64, 256, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # 16^2 x 128
            ResidualStack(128, 32, 128, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # 32^2 x 64
            ResidualStack(64, 16, 64, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            # 64^2 x 32
            ResidualStack(32, 8, 32, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            # 128^2 x 16
            ResidualStack(16, 8, 16, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
        )

    def forward(self, x):
        return self.block(x)


class ResNoise(nn.Module):
    def __init__(self):
        super(ResNoise, self).__init__()
        self.en = Encoder()
        self.de = Decoder()

    def forward(self, x):
        x = self.en(x)
        x = self.de(x)
        return x


class ResConnNoise(nn.Module):
    def __init__(self):
        super(ResConnNoise, self).__init__()
        self.en1 = nn.Sequential(
            # 256^2 x 3
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, 2, 1),
            ResidualStack(16, 8, 16, 2),
            nn.ReLU(True)
            # 128^2 x 16
        )
        self.en2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            ResidualStack(32, 8, 32, 2),
            nn.ReLU(True),
            # 64^2 x 32
        )
        self.en3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            ResidualStack(64, 16, 64, 2),
            nn.ReLU(),
            # 32^2 x 64
        )
        self.en4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            ResidualStack(128, 32, 128, 2),
            nn.ReLU(),
            # 16^2 x 128
        )
        self.en5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            ResidualStack(256, 64, 256, 2),
            nn.ReLU(),
            # 8^2 x 256
        )
        self.de1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            ResidualStack(256, 64, 256, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            # 16^2 x 128
        )
        self.de2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            ResidualStack(128, 32, 128, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
        )
        self.de3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            ResidualStack(64, 16, 64, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
        )
        self.de4 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            ResidualStack(32, 8, 32, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(True),
        )
        self.de5 = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            ResidualStack(16, 8, 16, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.ReLU(True),
        )

    def forward(self, x):
        en1 = self.en1(x)
        en2 = self.en2(en1)
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        en5 = self.en5(en4)

        de1 = self.de1(en5)
        de2 = self.de2(torch.cat((de1, en4), dim=1))
        de3 = self.de3(torch.cat((de2, en3), dim=1))
        de4 = self.de4(torch.cat((de3, en2), dim=1))
        de5 = self.de5(torch.cat((de4, en1), dim=1))
        return de5
