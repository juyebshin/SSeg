import os
import sys
# import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms # seg fault

import network
from optimizer import restore_snapshot
# from datasets import cityscapes
from datasets import kitti # seg fault
from config import assert_and_infer_cfg


class SemSeg:
    """
    """
    def __init__(self):
        print('Initializing Semantic Segmentation network...')
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # ROOT_DIR = './src/python/'
    ROOT_DIR = "./"
    print(ROOT_DIR)

    # Directory to pretrained model
    MODEL_DIR = os.path.join(ROOT_DIR, "pretrained_model/kitti_best.pth")
    print(MODEL_DIR)
    assert_and_infer_cfg('', train_mode=False)
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    # parser = argparse.ArgumentParser(description='test')
    # parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
    # parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
    # args = parser.parse_args()

    # get net
    # args.dataset_cls = cityscapes
    dataset_cls = kitti
    net = network.get_net('network.deepv3.DeepWV3Plus', dataset_cls, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=MODEL_DIR, restore_optimizer_bool=False)
    net.eval()
    print('Net restored.')

    # get data
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

    def getSemSeg(self, image_name, save_dir):
        img = Image.open(image_name).convert('RGB')
        img_tensor = self.img_transform(img)

        # predict
        with torch.no_grad():
            img = img_tensor.unsqueeze(0).cuda()
            pred = self.net(img)
            print('Inference done.')

        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        colorized = self.dataset_cls.colorize_mask(pred)
        # colorized.save(os.path.join(args.save_dir, 'color_mask.png'))
        # Added 2021-06-07 15:41
        colorizedArray = np.array(colorized.convert('RGB'))[...,::-1]
        # colorizedArray = colorizedArray[...,::-1]
        print('colorizedArray: ', colorizedArray.shape)
        cv2.imwrite(os.path.join(save_dir, 'color_mask.png'), colorizedArray)
        # save colorized predictions overlapped on original images
        # overlap = cv2.addWeighted(np.array(img), 0.5, colorizedArray, 0.5, 0)
        # cv2.imwrite(os.path.join(args.save_dir, 'overlap_mask.png'), overlap[:, :, ::-1])
        return colorizedArray

# for test
# parser = argparse.ArgumentParser(description='demo')
# parser.add_argument('--demo-image', type=str, default='', help='path to demo image', required=True)
# parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
# args = parser.parse_args()
# args.dataset_cls = kitti

# semseg = SemSeg
# result = semseg.getSemSeg(semseg, '000000.png', semseg.ROOT_DIR)

# Original
# parser = argparse.ArgumentParser(description='demo')
# parser.add_argument('--demo-image', type=str, default='', help='path to demo image', required=True)
# parser.add_argument('--snapshot', type=str, default='./pretrained_models/cityscapes_best.pth', help='pre-trained checkpoint', required=True)
# parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus', help='network architecture used for inference')
# parser.add_argument('--save-dir', type=str, default='./save', help='path to save your results')
# args = parser.parse_args()
# assert_and_infer_cfg('', train_mode=False)
# cudnn.benchmark = False
# torch.cuda.empty_cache()

# # get net
# # args.dataset_cls = cityscapes
# args.dataset_cls = kitti
# net = network.get_net(args, criterion=None)
# net = torch.nn.DataParallel(net).cuda()
# print('Net built.')
# net, _ = restore_snapshot(net, optimizer=None, snapshot=args.snapshot, restore_optimizer_bool=False)
# net.eval()
# print('Net restored.')

# # get data
# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
# img = Image.open(args.demo_image).convert('RGB')
# img_tensor = img_transform(img)

# # predict
# with torch.no_grad():
#     img = img_tensor.unsqueeze(0).cuda()
#     pred = net(img)
#     print('Inference done.')

# pred = pred.cpu().numpy().squeeze()
# pred = np.argmax(pred, axis=0)

# if not os.path.exists(args.save_dir):
#     os.makedirs(args.save_dir)

# colorized = args.dataset_cls.colorize_mask(pred)
# # colorized.save(os.path.join(args.save_dir, 'color_mask.png'))
# # Added 2021-06-07 15:41
# colorizedArray = np.array(colorized.convert('RGB'))[...,::-1]
# # colorizedArray = colorizedArray[...,::-1]
# print('colorizedArray: ', colorizedArray.shape)
# cv2.imwrite(os.path.join(args.save_dir, 'color_mask.png'), colorizedArray)
# # save colorized predictions overlapped on original images
# # overlap = cv2.addWeighted(np.array(img), 0.5, colorizedArray, 0.5, 0)
# # cv2.imwrite(os.path.join(args.save_dir, 'overlap_mask.png'), overlap[:, :, ::-1])

# label_out = np.zeros_like(pred)
# for label_id, train_id in args.dataset_cls.id_to_trainid.items():
#     label_out[np.where(pred == train_id)] = label_id
# cv2.imwrite(os.path.join(args.save_dir, 'pred_mask.png'), label_out)
# print('label: ', label_out.shape)
# print('Results saved.')
