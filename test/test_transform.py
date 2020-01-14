# -*- coding: utf-8 -*-

from transform import RandomTextOverlay, RandomCrop, RandomGaussianNoise
from dataset import PairDataset

import cv2
import os

dr_root = '/media/piston/data/Noise2Noise/'
pr = RandomCrop(256, pad_if_needed=True)
# ts = RandomTextOverlay(p=1, max_occupancy=50, length=(15, 30))
ts = RandomGaussianNoise(p=0.95, mean=0, std=25, fixed_distribution=False)
train_set = PairDataset(root_dir=os.path.join(dr_root, 'train'), pre_transform=pr,
                        source_transform=ts, target_transform=ts)

for s, t in train_set:
    print(s.shape, t.shape)
    cv2.imshow('s', s)
    cv2.imshow('t', t)
    cv2.waitKey(0)


def split(root_dir, target_dir):
    import os
    import glob
    import random
    import shutil

    imgs = glob.glob(os.path.join(root_dir, '*.JPEG'))
    simgs = random.sample(imgs, 10000)
    for img in simgs:
        file_name = img.split('/')[-1]
        shutil.move(img, os.path.join(target_dir, file_name))
