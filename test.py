#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet
from tqdm.autonotebook import tqdm
import torch
import os
from utils.utils import *
from torch.utils.data import DataLoader
from face_dataset import FaceMask
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results', imspth=[]):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    ims = np.array(im)
    # 逆归一化
    mean = np.tile(np.array([[0.485, 0.456, 0.406]]), [ims.shape[0], 1])
    std = np.tile(np.array([[0.229, 0.224, 0.225]]), [ims.shape[0], 1])
    mean = np.reshape(mean, newshape=[ims.shape[0], 3, 1, 1])  # 扩充维度以触发广播机制
    std = np.reshape(std, newshape=[ims.shape[0], 3, 1, 1])
    ims = (ims*std + mean)*255

    ims = np.transpose(ims, (0, 2, 3, 1))  # color channel 放到最后一维
    vis_ims = ims.copy().astype(np.uint8)
    vis_parsing_annos = parsing_anno.copy().astype(np.uint8)

    for i in range(vis_parsing_annos.shape[0]):
        vis_parsing_anno = cv2.resize(vis_parsing_annos[i], None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # use addWeighted(src1, alpha, src2, gamma) get dst = src1*alpha + src2*beta + gamma;
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_ims[i], cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        # Save result or not
        if save_im:
            sv_path = os.path.join('/home/data2/miles/face_parsing', save_path)
            if not os.path.exists(sv_path):
                os.makedirs(sv_path)
            cv2.imwrite(os.path.join(sv_path, imspth[i]), vis_parsing_anno)
            cv2.imwrite(os.path.join(sv_path, 'color_'+imspth[i]), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # return vis_im


def testval(model_path, sv_dir='res', sv_pred=False):
    n_classes = 19
    confusion_matrix = np.zeros((n_classes, n_classes)) # num_classes x num_classes
    cropsize = [448, 448]
    data_root = '/home/data2/DATASET/CelebAMask-HQ/'
    batch_size = 4
    ds = FaceMask(data_root, cropsize=cropsize, mode='test')
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                    )

    with torch.no_grad():
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()
        for iter, data in enumerate(dl):
            im = data['img']
            lb = data['label']
            impth = data['impth']
            lb = torch.squeeze(lb, 1)
            im = im.cuda()
            lb = lb.cuda()
            out = net(im)[0]
            size = lb.size()
            pred = out.cpu().numpy().argmax(1)
            gt = lb.cpu().numpy()
            vis_parsing_maps(im.cpu(), pred, stride=1, save_im=True, save_path='res', imspth=impth)
            vis_parsing_maps(im.cpu(), gt, stride=1, save_im=True, save_path='res_gt', imspth=impth)
            confusion_matrix += get_confusion_matrix(lb, out, size, n_classes, ignore=-1)  # [16, 19, 448, 448]

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                # cv2.imwrite(sv_path+'/', ) add imname

            if iter % 5 == 0:
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                pixel_acc = tp.sum() / pos.sum()
                mean_acc = (tp / np.maximum(1.0, pos)).mean()
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                print('index/allimg {}/{}. mean_IoU: {:1.5f}. pixel_acc: {:1.5f}. mean_acc: {:1.5f}.'.format(
                        iter*batch_size, len(dl)*batch_size, mean_IoU, pixel_acc, mean_acc))
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum() / pos.sum()
        mean_acc = (tp / np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        print('mean_IoU: {:1.5f}. pixel_acc: {:1.5f}. mean_acc: {:1.5f}.'.format(
               mean_IoU,  pixel_acc, mean_acc))
        return mean_IoU, IoU_array, pixel_acc, mean_acc


def evaluate(respth='./logs/CelebAMask', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join(respth, cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]  # shape [1, 19, 512, 512]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)  # shape [512, 512]
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))







if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '8'
    # evaluate(dspth='/home/data2/DATASET/CelebAMask-HQ/CelebA-HQ-img', cp='Bisenet_13_11600.pth')
    testval(model_path='/home/data2/miles/face_parsing/logs/CelebAMask/Bisenet_13_11600.pth')


