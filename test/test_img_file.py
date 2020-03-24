# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
import os
from torch.nn import functional as F
from model import *
from torch import nn
from dataset import PairDataset
from transform import RandomCrop, RandomTextOverlay
from torchtoolbox import transform


class wapper(nn.Module):
    def __init__(self, module):
        super(wapper, self).__init__()
        self.module = module


model = UNet()
# model = ResConnNoise()
# model = nn.DataParallel(model)
model = wapper(model)
model.load_state_dict(torch.load('../param/text_denoise_99.pt', map_location=torch.device('cpu')))
model = model.module.cpu()

test_transform = transform.Compose([
    transform.Pad((0, 4, 0, 4)),
    transform.ToTensor(),
])


def get_psnr(input, target):
    """Computes peak signal-to-noise ratio."""

    return 10 * torch.log10(1 / F.mse_loss(input, target))


def to_numpy(tensor):
    tensor = tensor.squeeze().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)[..., ::-1]
    return tensor


with torch.no_grad():
    s = cv2.imread('/media/piston/yx/hiddenWatermarkedImg/win.bmp')[..., ::-1]
    s = test_transform(s).unsqueeze(0)
    print(s.shape)
    outputs = model(s)
    outputs = outputs.clamp(0, 1)
    outputs = to_numpy(outputs)
    outputs = outputs[4:-4, :, :]
    cv2.imshow('Output', outputs)
    cv2.imwrite('/media/piston/yx/hiddenWatermarkedImg/win_denoise.bmp', outputs)
    cv2.waitKey(0)
