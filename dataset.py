# -*- coding: utf-8 -*-
import glob
import os
from torch.utils.data import Dataset
from torchtoolbox.data import cv_loader


class PairDataset(Dataset):
    def __init__(self, root_dir, pre_transform=None, source_transform=None,
                 target_transform=None, post_transform=None, loader=cv_loader):
        self.items = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))]
        self.pre_transform = pre_transform
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.post_transform = post_transform
        self.loader = loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        img = self.loader(self.items[item])
        if self.pre_transform:
            img = self.pre_transform(img)

        source = img.copy()
        if self.source_transform:
            source = self.source_transform(source)

        target = img.copy()
        if self.target_transform:
            target = self.target_transform(target)

        if self.post_transform:
            source = self.post_transform(source)
            target = self.post_transform(target)

        return source, target
