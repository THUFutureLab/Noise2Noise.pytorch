# # -*- coding: utf-8 -*-

import PIL.Image
import numpy as np
import torch
import cv2
import pickle


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


# img = np.asarray(PIL.Image.open('/home/piston/Downloads/IXI_DATA/MRI/IXI002-Guys-0828-T1_025.png'), dtype=np.uint8)
# img = img.astype(np.float32) / 255.0 - 0.5
# img = img[:255, :255]
# x = np.zeros_like(img)
# trans = np.random.randint(-64, 64 + 1, size=(2,))
# x[trans[0], trans[1]] = 1.0
# xfft = fftshift2d(np.fft.fft2(x).astype(np.complex64))
#
# img = np.roll(img, trans, axis=(0, 1))
# spec = np.fft.fft2(img).astype(np.complex64)
#
# print(np.sum(xfft))
# sx = xfft * spec
# sx = np.abs(sx)
# sx = sx / np.max(sx) * 255.
#
# d0 = np.abs(spec)
# d3 = d0 / np.max(d0) * 255.
#
# d0 = fftshift2d(d3.copy())
# # d0 = np.roll(d0, np.random.randint(-64, 65), axis=(0, 1))
# print(np.sum(sx-d3))
# cv2.imshow('1', d0)
# cv2.imshow('2', sx)
# cv2.imshow('3', d3)
# cv2.waitKey(0)

from MRI.utils import MRI_Dataset, mri_collate, as_complex
from torch.utils.data import DataLoader


def fftshift3d(x, ifft):
    assert len(x.shape) == 5
    s0 = (x.shape[2] // 2) + (0 if ifft else 1)
    s1 = (x.shape[3] // 2) + (0 if ifft else 1)
    x = torch.cat([x[:, :, s0:, :, :], x[:, :, :s0, :, :]], dim=2)
    x = torch.cat([x[:, :, :, s1:, :], x[:, :, :, :s1, :]], dim=3)
    return x


dt = MRI_Dataset('/home/piston/Downloads/IXI_DATA/pkl_data/ixi_valid.pkl')
data = DataLoader(dt, shuffle=False, batch_size=4, collate_fn=mri_collate)

for i, (inp, target, spec_val, spec_mask, sm, sp) in enumerate(data):
    print(inp.shape, target.shape, spec_val.shape, spec_mask.shape)
    print(inp.dtype, target.dtype, spec_val.dtype, spec_mask.dtype)
    # f = torch.fft()
    # cv2.imshow('inp', inp)
    # cv2.imshow('tar', target)
    # cv2.imshow('sm', spec_mask)
    # cv2.waitKey(0)
    inp = inp.unsqueeze(-1)
    in_fft = torch.cat([inp, torch.zeros_like(inp)], dim=-1)
    ift = torch.fft(in_fft, signal_ndim=2)
    ift = fftshift3d(ift, False)
    print(ift.shape, ift.dtype)
    ift_np = ift.numpy()
    ift_np = ift_np[..., 0] + 1j * ift_np[..., 1]
    print(ift_np.dtype, ift_np.shape)
    d0 = np.abs(ift_np)
    d3 = d0 / np.max(d0) * 255.

    cv2.imshow('1', inp[0, 0, :, :].numpy())
    cv2.imshow('tar', target[0, 0, :, :].numpy())
    cv2.imshow('sm', sm[0, 0, :, :].numpy())
    cv2.imshow('0', d3[0, 0, :, :])
    cv2.imwrite('/home/piston/Downloads/IXI_DATA/MRI/1.bmp', inp[0, 0, :, :].numpy())
    cv2.imwrite('/home/piston/Downloads/IXI_DATA/MRI/tar.bmp', target[0, 0, :, :].numpy())
    cv2.imwrite('/home/piston/Downloads/IXI_DATA/MRI/sm.bmp', sm[0, 0, :, :].numpy())
    cv2.imwrite('/home/piston/Downloads/IXI_DATA/MRI/0.bmp', d3[0, 0, :, :].astype('uint8'))
    cv2.waitKey(0)
    break


# d = torch.randn(4, 1, 255, 255)
# f = torch.fft(d, signal_ndim=2)
# print(f.shape)


# def post_process(denoised: torch.FloatTensor, spec_mask_var: torch.FloatTensor, spec_value_var):
#     denoised, spec_mask_var = map(to_numpy, (denoised, spec_mask_var))
#     denoised_spec = np.fft.fft2(denoised)
#     denoised_spec = fftshift3d(denoised_spec, False)
#     spec_mask_c64 = spec_mask_var.astype(np.complex64)
#     denoised_spec = spec_value_var * spec_mask_c64 + denoised_spec * (1. - spec_mask_c64)
#     denoised = np.fft.ifft2(fftshift3d(denoised_spec, True)).astype(np.float32)
#     return torch.as_tensor(denoised)
