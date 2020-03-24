# -*- coding: utf-8 -*-


import numpy as np
import math
import os
import time
import torch
import argparse
from model import *
from torch import optim
from torch import nn
from MRI.utils import *
from torchtoolbox.tools import check_dir
from torchtoolbox.metric import NumericalCost
from torch.utils.data import Dataset, DataLoader
from MRI.config.test_load_json import ArgsParse

args = ArgsParse('./config/train_config.json')
print(args)


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


check_dir(args.save_dir)
device = torch.device("cuda:0")
device_ids = args.devices
device_ids = [int(device) for device in device_ids]

lr = args.lr
train_loss = args.loss
epochs = args.epochs
num_workers = args.num_workers
batch_size = args.batch_size
rampup_length, rampdown_length = args.rampup_length, args.rampdown_length
adam_param = tuple(map(float, args.adam_param))

train_set = MRI_Dataset(os.path.join(args.data_path, 'ixi_train.pkl'), args.p_at_edge, args.roll, args.t,
                        args.corrupt_targets)
val_set = MRI_Dataset(os.path.join(args.data_path, 'ixi_valid.pkl'), args.p_at_edge, args.roll, args.t)
train_data = DataLoader(train_set, batch_size, True, num_workers=num_workers, collate_fn=mri_collate, drop_last=True)
val_data = DataLoader(val_set, batch_size, False, num_workers=num_workers, collate_fn=mri_collate, drop_last=False)

model = UNet(in_channels=1, out_channels=1, act_type='lrelu').cuda()

Loss = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=adam_param[2],
                       weight_decay=args.wd)

psnr_record = NumericalCost(name='Avg PSNR')
loss_record = NumericalCost(name='Loss')


def adjust_learning_rate(optimizer, lr, betas):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = betas


def train():
    for epoch in range(epochs):
        loss_record.reset()
        tic = time.time()

        rampup_value = rampup(epoch, rampup_length)
        rampdown_value = rampdown(epoch, epochs, rampdown_length)

        adam_beta1 = (rampdown_value * 0.9) + ((1.0 - rampdown_value) * 0.5)
        learning_rate = rampup_value * rampdown_value * lr

        adjust_learning_rate(optimizer, learning_rate, (adam_beta1, 0.999))

        model.train()
        for i, (source, target, spec_val, spec_mask) in enumerate(train_data):
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            spec_val = spec_val.to(device, non_blocking=True)
            spec_mask = spec_mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            denoised = model(source)[:, :, :-1, :-1]
            denoised = post_process(denoised, spec_mask, spec_val)

            loss = Loss(denoised, target)
            loss.backward()
            optimizer.step()

            loss_record.update(loss)
            if i % args.log_interval == 0 and i != 0:
                print('Epoch {}, Iter {}, {}:{:.5}, {} samples/s.'.format(
                    epoch, i, loss_record.name, loss_record.get(),
                    int((i * batch_size) // (time.time() - tic))
                ))

        train_speed = int(len(train_set) // (time.time() - tic))
        print('Epoch {}, {}:{:.5}, {} samples/s, lr: {:.5}'.format(
            epoch, loss_record.name, loss_record.get(), train_speed, learning_rate))
        test(epoch)


def test(epoch):
    model.eval()
    psnr_record.reset()
    loss_record.reset()

    for source, target, spec_val, spec_mask in val_data:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        spec_val = spec_val.to(device, non_blocking=True)
        spec_mask = spec_mask.to(device, non_blocking=True)

        denoised = model(source)[:, :, :-1, :-1]
        denoised = post_process(denoised, spec_mask, spec_val)

        denoised = torch.clamp(denoised, -0.5, 0.5)

        loss = Loss(denoised, target)

        for b in range(source.size()[0]):
            psnr_record.update(get_psnr(denoised[b], target[b]))
        loss_record.update(loss)

    print('Epoch {}, {}:{:.5}, {}:{:.5}\n'.format(
        epoch, psnr_record.name, psnr_record.get(),
        loss_record.name, loss_record.get()))


if __name__ == '__main__':
    train()
