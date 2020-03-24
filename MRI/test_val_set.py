# -*- coding: utf-8 -*-

from MRI.utils import MRI_Dataset, mri_collate, as_complex, get_psnr
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

val_set = MRI_Dataset('/home/piston/Downloads/IXI_DATA/pkl_data/ixi_valid.pkl')
val_data = DataLoader(val_set, 2, False, collate_fn=mri_collate, drop_last=False)


def fftshift3d(x, ifft):
    assert len(x.shape) == 5
    s0 = (x.shape[2] // 2) + (0 if ifft else 1)
    s1 = (x.shape[3] // 2) + (0 if ifft else 1)
    x = torch.cat([x[:, :, s0:, :, :], x[:, :, :s0, :, :]], dim=2)
    x = torch.cat([x[:, :, :, s1:, :], x[:, :, :, :s1, :]], dim=3)
    return x


def post_process(denoised, spec_mask_var, spec_value_var):
    denoised = as_complex(denoised)
    denoised_spec = torch.fft(denoised, signal_ndim=2)
    denoised_spec = fftshift3d(denoised_spec, False)
    spec_mask_c64 = as_complex(spec_mask_var)
    denoised_spec = spec_value_var * spec_mask_c64 + denoised_spec * (1. - spec_mask_c64)
    denoised = torch.ifft(fftshift3d(denoised_spec, True), signal_ndim=2)[..., 0]
    return denoised


if __name__ == '__main__':
    for source, target, spec_val, spec_mask in val_data:
        source = source[:, :, :-1, :-1]
        print(source.shape, target.shape, spec_val.shape, spec_mask.shape)
        cv2.imshow('source', source[0].squeeze().numpy())
        print(torch.min(source))
        ps_s = post_process(source, spec_mask, spec_val)
        print(torch.min(ps_s))
        cv2.imshow('ps_s', ps_s[0].squeeze().numpy())
        print(torch.min(target))
        cv2.imshow('target', target[0].squeeze().numpy())
        cv2.waitKey(0)


        break
