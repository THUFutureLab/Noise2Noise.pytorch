# -*- coding: utf-8 -*-

from transform import *
from model import *
from dataset import PairDataset
from torch.utils.data import DataLoader
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.metric import NumericalCost
from torchtoolbox import transform
from torchtoolbox.tools import check_dir
from torch import optim
from torch import nn
from loss import HDRLoss, L0Loss
from torch.nn import functional as F

import argparse
import torch
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='Train a Noise2Noise model')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--devices', type=str, default='0',
                    help='gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0,
                    help='learning rate. default is 0.')
parser.add_argument('--adam-param', type=str, default='0.9,0.99,1e-8',
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--loss', type=str, choices=['l0', 'l1', 'l2', 'hdr'], default='l1',
                    help='loss func.')
# parser.add_argument('--model', type=str, required=True,
#                     help='type of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=int, default=256,
                    help='size of the input image size. default is 224')
parser.add_argument('-n', '--noise-type', help='noise type',
                    choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                    help='name of training log file')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

check_dir(args.save_dir)
device = torch.device("cuda:0")
device_ids = args.devices.strip().split(',')
device_ids = [int(device) for device in device_ids]

lr = args.lr
train_loss = args.loss
epochs = args.epochs
num_workers = args.num_workers
batch_size = args.batch_size * len(device_ids)
adam_param = tuple(map(float, args.adam_param.split(',')))

pre_transform = RandomCrop(args.input_size, pad_if_needed=True)

source_transform = transform.Compose([
    # RandomGaussianNoise(p=0.95, mean=0, std=25, fixed_distribution=False),
    RandomTextOverlay(p=1, max_occupancy=30, length=(15, 30)),
    transform.ToTensor(),
])

target_transform = transform.Compose([
    # RandomGaussianNoise(p=0.95, mean=0, std=25, fixed_distribution=False),
    RandomTextOverlay(p=1, max_occupancy=30, length=(15, 30)),
    transform.ToTensor(),
])

test_transform = transform.ToTensor()

train_set = PairDataset(root_dir=os.path.join(args.data_path, 'train'), pre_transform=pre_transform,
                        source_transform=source_transform, target_transform=target_transform)
test_set = PairDataset(root_dir=os.path.join(args.data_path, 'test'), pre_transform=pre_transform,
                       source_transform=source_transform, target_transform=test_transform)

train_data = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
test_data = DataLoader(test_set, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False)

model = UNet()
# model = ResNoise()
KaimingInitializer(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_param[0:2], eps=adam_param[2],
                       weight_decay=args.wd)

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=epochs / 5, factor=0.5, verbose=True)
model = nn.DataParallel(model)

if train_loss == 'hdr':
    Loss = HDRLoss()
elif train_loss == 'l0':
    Loss = L0Loss()
elif train_loss == 'l1':
    Loss = nn.L1Loss()
else:
    Loss = nn.MSELoss()

psnr_record = NumericalCost(name='Avg PSNR')
loss_record = NumericalCost(name='Loss')


def get_psnr(input, target):
    """Computes peak signal-to-noise ratio."""

    return 10 * torch.log10(1 / F.mse_loss(input, target))


@torch.no_grad()
def test(epoch=0, save_status=True):
    psnr_record.reset()
    loss_record.reset()
    model.eval()
    for i, (source, target) in enumerate(test_data):

        source = source.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        outputs = model(source).cpu()
        outputs = outputs.clamp(0, 1)
        loss = Loss(outputs, target)
        loss_record.update(loss)

        for b in range(source.size()[0]):
            psnr_record.update(get_psnr(outputs[b], target[b]))

    print('Epoch {}, {}:{:.5}, {}:{:.5}'.format(
        epoch, psnr_record.name, psnr_record.get(),
        loss_record.name, loss_record.get()))
    lr_scheduler.step(loss_record.get())
    if save_status:
        torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(epoch)))
        print("Finish save stats.")


def train():
    for epoch in range(epochs):
        psnr_record.reset()
        loss_record.reset()
        tic = time.time()

        model.train()
        for i, (source, target) in enumerate(train_data):
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(source)

            loss = Loss(outputs, target)
            loss.backward()
            optimizer.step()

            loss_record.update(loss)
            if i % args.log_interval == 0 and i != 0:
                print('Epoch {}, Iter {}, {}:{:.5}, {} samples/s.'.format(
                    epoch, i, loss_record.name, loss_record.get(),
                    int((i * batch_size) // (time.time() - tic))
                ))
        if train_loss == 'l0':
            Loss.gamma = 2 * (1 - epoch / epochs)

        train_speed = int(len(train_set) // (time.time() - tic))
        print('Epoch {}, {}:{:.5}, {} samples/s.'.format(
            epoch, loss_record.name, loss_record.get(), train_speed))
        test(epoch, True)


if __name__ == '__main__':
    train()
