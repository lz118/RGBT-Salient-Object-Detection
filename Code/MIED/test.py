coding='utf-8'
import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from net import Mnet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
if __name__ == '__main__':
    model_path='./model/normal.pth'
    out_path = './output'
    data  = Data(root='path/to/test_images',mode='test')
    loader = DataLoader(data, batch_size=1,shuffle=False)
    net = Mnet().cuda()
    print('loading model from %s...' % model_path)
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    time_s = time.time()
    img_num = len(loader)
    net.eval()
    with torch.no_grad():
        for rgb, t, _ , (H, W), name in loader:
            print(name[0])
            score, score1, score2,score_g = net(rgb.cuda().float(), t.cuda().float())
            score = F.interpolate(score, size=(H, W), mode='bilinear')
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            multi_fuse = 255 * pred
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), multi_fuse)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))



