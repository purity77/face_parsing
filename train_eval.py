#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
import numpy as np
from optimizer import Optimizer
import torch
import traceback
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
from utils.utils import get_last_weights
from torch.utils.data import DataLoader
import os
import datetime
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3, 4'


def get_args():
    parser = argparse.ArgumentParser('face parsing by - miles')
    parser.add_argument('-p', '--project', type=str, default='flickr27', help='project file that contains parameters')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=200, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement '
                             'after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, '
                             'set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')
    args = parser.parse_args()
    return args


def save_checkpoint(model, name):
    torch.save(model.module.state_dict(), os.path.join(opt.saved_path, name))


def train(opt):
    opt.saved_path = opt.saved_path + 'CelebAMask'
    opt.opt_path = opt.log_path + 'CelebAMask'+'tensorboard'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    # dataset
    gpu_number = torch.cuda.device_count()
    n_classes = 19
    n_img_all_gpu = opt.batch_size * gpu_number
    cropsize = [448, 448]
    data_root = '/home/data2/DATASET/CelebAMask-HQ/'
    num_workers = 8

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    dl = DataLoader(ds,
                    batch_size=n_img_all_gpu,
                    shuffle=True,
                    num_workers=num_workers
                    )
    ds_eval = FaceMask(data_root, cropsize=cropsize, mode='val')
    dl_eval = DataLoader(ds_eval,
                         batch_size=n_img_all_gpu,
                         shuffle=True,
                         num_workers=num_workers
                         )

    ignore_idx = -100
    net = BiSeNet(n_classes=n_classes)
    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            # ??
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0
        try:
            ret = net.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights '
                'with different number of classes. The rest of the weights should be loaded already.')

            print(
                f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    net = net.cuda()
    net = nn.DataParallel(net)

    score_thres = 0.7
    n_min = n_img_all_gpu * cropsize[0] * cropsize[1] // opt.batch_size  # ??
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = opt.lr
    max_iter = 80000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
        model=net.module,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim.optim, patience=3, verbose=True)
    # train loop
    loss_avg = []
    step = max(0, last_step)
    max_iter = len(dl)
    best_epoch = 0
    epoch = 0
    best_loss = 1e5
    net.train()
    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // max_iter
            if epoch < last_epoch:
                continue
            epoch_loss = []
            progress_bar = tqdm(dl)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * max_iter:
                    progress_bar.update()
                    continue
                try:
                    im = data['img']
                    lb = data['label']
                    lb = torch.squeeze(lb, 1)
                    im = im.cuda()
                    lb = lb.cuda()

                    optim.zero_grad()
                    out, out16, out32 = net(im)
                    lossp = LossP(out, lb)
                    loss2 = Loss2(out16, lb)
                    loss3 = Loss3(out32, lb)
                    loss = lossp + loss2 + loss3
                    if loss == 0 or not torch.isfinite(loss):
                        continue
                    loss.backward()
                    optim.step()
                    loss_avg.append(loss.item())
                    #  print training log message
                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. p_loss: {:.5f}. 2_loss: {:.5f}.'.format(
                            step, epoch, opt.num_epochs, iter + 1, max_iter, lossp.item(),
                            loss2.item()))
                    progress_bar.set_description('2_loss: {:.5f}. 3_loss: {:.5f}. loss_avg: {:.5f}'.format(
                        loss2.item(), loss3.item(), loss.item()))
                    writer.add_scalars('Lossp', {'train': lossp}, step)
                    writer.add_scalars('loss2', {'train': loss2}, step)
                    writer.add_scalars('loss3', {'train': loss3}, step)
                    writer.add_scalars('loss_avg', {'train': loss}, step)

                    # log learning_rate
                    lr = optim.lr
                    writer.add_scalar('learning_rate', lr, step)
                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(net, f'Bisenet_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Erro]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                net.eval()
                loss_p = []
                loss_2 = []
                loss_3 = []
                for iter, data in enumerate(dl_eval):
                    with torch.no_grad():
                        im = data['img']
                        lb = data['label']
                        lb = torch.squeeze(lb, 1)
                        im = im.cuda()
                        lb = lb.cuda()

                        out, out16, out32 = net(im)
                        lossp = LossP(out, lb)
                        loss2 = Loss2(out16, lb)
                        loss3 = Loss3(out32, lb)
                        loss = lossp + loss2 + loss3
                        if loss == 0 or not torch.isfinite(loss):
                            continue
                        loss_p.append(lossp.item())
                        loss_2.append(loss2.item())
                        loss_3.append(loss3.item())
                lossp = np.mean(loss_p)
                loss2 = np.mean(loss_2)
                loss3 = np.mean(loss_3)
                loss = lossp + loss2 + loss3
                print(
                    'Val. Epoch: {}/{}. p_loss: {:1.5f}. 2_loss: {:1.5f}. 3_loss: {:1.5f}. Total_loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, lossp, loss2, loss3, loss))
                writer.add_scalars('Total_loss', {'val': loss}, step)
                writer.add_scalars('p_loss', {'val': lossp}, step)
                writer.add_scalars('2_loss', {'val': loss2}, step)
                writer.add_scalars('2_loss', {'val': loss3}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(net, f'Bisenet_{epoch}_{step}.pth')

                net.train()  # ??
                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(net, f'Bisenet_{epoch}_{step}.pth')
        writer.close()
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
