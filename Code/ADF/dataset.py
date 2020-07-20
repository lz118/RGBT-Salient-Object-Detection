import os
from PIL import Image
import cv2
import torch
from torch.utils import data
import numpy as np
import random


dataset_mean_rgb=np.array([140.72236226, 157.80380783, 135.71176741]) #mean value of trainset rgb
dataset_mean_t=np.array([87.13143683, 92.03255377, 192.20629105])  #mean value of trainset t

class ImageDataTrain(data.Dataset):
    def __init__(self, train_root):
        self.data_rgb_root = os.path.join(train_root,"RGB")
        self.data_t_root = os.path.join(train_root,"T")
        self.gt_root=os.path.join(train_root,"GT")
        self.imlist=os.listdir(self.data_rgb_root)
        self.im_num = len(self.imlist)
    def __getitem__(self, item):
        im_name=self.imlist[item % self.im_num]
        rgb_img = load_image(os.path.join(self.data_rgb_root, im_name),mode='rgb')
        t_img = load_image(os.path.join(self.data_t_root, im_name),mode='t')
        label=load_sal_label(os.path.join(self.gt_root, im_name[:-4]+'.png'))
        rgb_img,t_img,label = cv_random_flip(rgb_img,t_img,label)
        rgb_img = torch.Tensor(rgb_img)
        t_img = torch.Tensor(t_img)
        label = torch.Tensor(label)
        sample = {'rgb_img': rgb_img, 't_img': t_img,'label':label}
        return sample

    def __len__(self):
        return self.im_num


class ImageDataTest(data.Dataset):
    def __init__(self,test_root):
        self.test_rgb_root = os.path.join(test_root,"RGB")
        self.test_t_root = os.path.join(test_root,"T")
        self.image_list = os.listdir(self.test_rgb_root)
        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        rgb_path=os.path.join(self.test_rgb_root, self.image_list[item])
        t_path=os.path.join(self.test_t_root, self.image_list[item])
        rgb,t,im_size = load_image_test(rgb_path,t_path)
        rgb = torch.Tensor(rgb)
        t = torch.Tensor(t)
        return {'rgb': rgb,'t':t, 'name': self.image_list[item % self.image_num], 'size': im_size}
    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root)
        data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=1, pin_memory=pin)
    return data_loader

def load_image(path,mode='rgb'):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    if mode=='rgb':
        in_ -= dataset_mean_rgb
    elif mode=='t':
        in_ -= dataset_mean_t
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(rgb_path,t_path):
    if not os.path.exists(rgb_path):
        print('File {} not exists'.format(rgb_path))
    if not os.path.exists(t_path):
        print('File {} not exists'.format(t_path))
    rgb,t = cv2.imread(rgb_path),cv2.imread(t_path)
    in_rgb,in_t = np.array(rgb, dtype=np.float32),np.array(t, dtype=np.float32)
    im_size = tuple(in_rgb.shape[:2])
    in_rgb -= dataset_mean_rgb
    in_t -= dataset_mean_t
    in_rgb,in_t = in_rgb.transpose((2,0,1)),in_t.transpose((2,0,1))
    return in_rgb,in_t,im_size

def load_sal_label(path):
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(rgb,t,label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        rgb = rgb[:,:,::-1].copy()
        t = t[:, :, ::-1].copy()
        label = label[:,:,::-1].copy()
    return rgb,t,label
