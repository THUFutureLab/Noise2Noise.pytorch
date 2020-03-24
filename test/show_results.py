# -*- coding: utf-8 -*-

import torch
import numpy as np
import cv2
import os
from torch.nn import functional as F
from model import *
from torch import nn
from dataset import PairDataset
from transform import RandomCrop, RandomTextOverlay, RandomGaussianNoise
from torchtoolbox import transform


class wapper(nn.Module):
    def __init__(self, module):
        super(wapper, self).__init__()
        self.module = module


model = UNet()
# model = ResConnNoise()
# model = nn.DataParallel(model)
model = wapper(model)
model.load_state_dict(torch.load('../param/5_gaussian.pt', map_location=torch.device('cpu')))
model = model.module.cpu()

pre_transform = RandomCrop(256, pad_if_needed=True)
source_transform = transform.Compose([
    RandomGaussianNoise(p=0.95, mean=0, std=25, fixed_distribution=False),
    # RandomTextOverlay(p=1, max_occupancy=30, length=(15, 30)),
    transform.ToTensor(),
])

test_transform = transform.ToTensor()
dt = PairDataset('/media/piston/data/Noise2Noise/test', pre_transform=pre_transform,
                 source_transform=source_transform, target_transform=test_transform)


def get_psnr(input, target):
    """Computes peak signal-to-noise ratio."""

    return 10 * torch.log10(1 / F.mse_loss(input, target))


def to_numpy(tensor):
    tensor = tensor.squeeze().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)[..., ::-1]
    return tensor


with torch.no_grad():
    for i, (s, t) in enumerate(dt):
        s, t = s.unsqueeze(0), t.unsqueeze(0)
        print('Noise PSNR:{}'.format(get_psnr(s, t)))
        outputs = model(s)
        outputs = outputs.clamp(0, 1)
        print('Denoise PSNR:{}'.format(get_psnr(outputs, t)))
        s, o, t = map(to_numpy, (s, outputs, t))

        cv2.imshow('Source', s)
        cv2.imshow('Denoise', o)
        cv2.imshow('Target', t)
        #
        cv2.imwrite('../save_imgs/{}_{}.bmp'.format(i, 'source'), s)
        cv2.imwrite('../save_imgs/{}_{}.bmp'.format(i, 'denoise'), o)
        cv2.imwrite('../save_imgs/{}_{}.bmp'.format(i, 'ground_truth'), t)
        cv2.waitKey(0)
