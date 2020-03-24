# -*- coding: utf-8 -*-
__all__ = ['RollData', 'CorruptData', 'MRI_Dataset', 'load_pkl', 'to_numpy', 'mri_collate',
           'rampup', 'rampdown', 'as_complex', 'get_psnr']

import pickle
import torch
import math
import numpy as np
import collections.abc as container_abcs
from torch.utils.data import Dataset
import torch.nn.functional as F


def as_complex(tensor):
    if isinstance(tensor, torch.Tensor):
        return torch.stack([tensor, torch.zeros_like(tensor)], dim=-1)
    elif isinstance(tensor, np.ndarray):
        if tensor.dtype == np.float32:
            tensor = torch.as_tensor(tensor)
            return as_complex(tensor)
        elif tensor.dtype == np.complex64:
            return torch.stack([torch.as_tensor(tensor.real), torch.as_tensor(tensor.imag)], dim=-1)
        else:
            return NotImplementedError
    elif isinstance(tensor, (tuple, list)):
        return [as_complex(t) for t in tensor]
    else:
        raise NotImplementedError


@torch.no_grad()
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif tensor.get_device() == -1:  # cpu tensor
        return tensor.numpy()
    else:
        return tensor.cpu().numpy()


def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


class RollData(object):
    def __init__(self, t=64):
        self.augment_translate_cache = dict()
        self.t = t

    def get_params(self, t, img):
        trans = np.random.randint(-t, t + 1, size=(2,))
        key = (trans[0], trans[1])
        if key not in self.augment_translate_cache:
            x = np.zeros_like(img)
            x[trans[0], trans[1]] = 1.0
            self.augment_translate_cache[key] = fftshift2d(np.fft.fft2(x).astype(np.complex64))
        return trans, key

    def __call__(self, img, spec):
        trans, key = self.get_params(self.t, img)
        img = np.roll(img, trans, axis=(0, 1))
        spec *= self.augment_translate_cache[key]
        return img, spec


class CorruptData(object):
    def __init__(self, p_at_edge=0.025):
        ctype = 'bspec'
        self.bernoulli_mask_cache = dict()
        self.p_at_edge = p_at_edge

    def get_param(self, spec):
        if self.bernoulli_mask_cache.get(self.p_at_edge) is None:
            h = [s // 2 for s in spec.shape]
            r = [np.arange(s, dtype=np.float32) - h for s, h in zip(spec.shape, h)]
            r = [x ** 2 for x in r]
            r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
            m = (self.p_at_edge ** (1. / h[1])) ** r
            self.bernoulli_mask_cache[self.p_at_edge] = m
            print('Bernoulli probability at edge = %.5f' % m[h[0], 0])
            print('Average Bernoulli probability = %.5f' % np.mean(m))
        mask = self.bernoulli_mask_cache[self.p_at_edge]
        return mask

    def __call__(self, img, spec):
        mask = self.get_param(spec)
        keep = (np.random.uniform(0.0, 1.0, size=spec.shape) ** 2 < mask)
        keep = keep & keep[::-1, ::-1]
        sval = spec * keep
        smsk = keep.astype(np.float32)
        spec = fftshift2d(sval / (mask + ~keep), ifft=True)  # Add 1.0 to not-kept values to prevent div-by-zero.
        img = np.real(np.fft.ifft2(spec)).astype(np.float32)
        return img, sval, smsk


class MRI_Dataset(Dataset):
    def __init__(self, pkl_path, p_at_edge=0.025, roll=True, t=64, corrupt_targets=False):
        img, spec = load_pkl(pkl_path)
        img = img[:, :-1, :-1]
        assert img.dtype == np.uint8
        img = img.astype(np.float32) / 255.0 - 0.5
        self.img = img
        self.spec = spec
        self.corrupt = CorruptData(p_at_edge)
        self.roll = RollData(t) if roll else None
        self.corrupt_targets = corrupt_targets

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, item):
        img = self.img[item]
        spec = self.spec[item]

        if self.roll is not None:
            img, spec = self.roll(img, spec)

        inp, spec_val, spec_mask = self.corrupt(img, spec)
        if self.corrupt_targets:
            target, _, _ = self.corrupt(img, spec)
        else:
            target = img.copy()
        inp = np.pad(inp, ((0, 1), (0, 1)), 'constant', constant_values=-.5)
        inp = np.expand_dims(inp, 0)
        target = np.expand_dims(target, 0)
        spec_val = np.expand_dims(spec_val, 0)
        spec_mask = np.expand_dims(spec_mask, 0)

        return inp, target, spec_val, spec_mask


def get_psnr(input, target):
    """Computes peak signal-to-noise ratio."""

    return 10 * torch.log10(1 / F.mse_loss(input, target))


def mri_collate(batch):
    elem = batch[0]
    if isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [mri_collate(samples) for samples in transposed]
    elif elem.dtype == np.float32:
        return torch.as_tensor(np.stack(batch, axis=0))
    elif elem.dtype == np.complex64:
        # return torch.stack(as_complex(batch), dim=0)
        return as_complex(np.stack(batch, axis=0))
    else:
        raise NotImplementedError


def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * 5.0)
    return 1.0


def rampdown(epoch, num_epochs, rampdown_length):
    if epoch >= (num_epochs - rampdown_length):
        ep = (epoch - (num_epochs - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    return 1.0
