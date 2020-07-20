#coding=utf-8

import os
import cv2
import numpy as np
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset

#BGR
mean_rgb = np.array([[[0.551*255, 0.619*255, 0.532*255]]])
mean_t =np.array([[[0.341*255,  0.360*255, 0.753*255]]])
std_rgb = np.array([[[0.241 * 255, 0.236 * 255, 0.244 * 255]]])
std_t = np.array([[[0.208 * 255, 0.269 * 255, 0.241 * 255]]])


class Data(Dataset):
    def __init__(self, root,mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            tpath = os.path.join(root, 'T', line[:-4]+'.jpg')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,maskpath])

        if mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,mean2=mean_t,std1=std_rgb,std2=std_t),
                                                transform.Resize(400,400),
                                                transform.RandomHorizontalFlip(),transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb,mean2=mean_t,std1=std_rgb,std2=std_t),
                                                transform.Resize(400,400),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath,tpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        t = cv2.imread(tpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        rgb,t,mask = self.transform(rgb,t,mask)
        return rgb,t,mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)