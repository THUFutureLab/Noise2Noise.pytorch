# -*- coding: utf-8 -*-
import glob
import os
from torch.utils.data import Dataset
from torchtoolbox.data import cv_loader


class PairDataset(Dataset):
    def __init__(self, root_dir, pre_transform, source_transform, target_transform, loader=cv_loader):
        self.items = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))]
        self.pre_transform = pre_transform
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        img = self.loader(self.items[item])
        if self.pre_transform:
            img = self.pre_transform(img)

        if self.source_transform:
            source = self.source_transform(img)
        else:
            source = img.copy()

        if self.target_transform:
            target = self.target_transform(img)
        else:
            target = img.copy()

        return source, target
