import torch
from collections import OrderedDict
from torch.nn import functional as F
from torch.optim import Adam
from APMFnet import build_model
from LaplaceEdge import Laplace_edge_layer
import numpy as np
import os
import cv2
import time

def train(train_loader,config):

    net = build_model(config)
    lap = Laplace_edge_layer()
    if config.cuda:
        net = net.cuda()
        lap = lap.cuda()

    print('loading pretrained model...')
    if config.cuda:
        stat = torch.load(config.pretrained_model)
    else:
        stat = torch.load(config.pretrained_model, map_location='cpu')
    net.rgb_net.load_pretrained_model(stat)
    net.t_net.load_pretrained_model(stat)
    newstat = OrderedDict()
    for i in stat:
        [idx, wtype] = i.split('.')
        if int(idx) >= 5:
            newstat[str(int(idx) - 4) + '.' + wtype] = stat[i]
    net.fusnet.base.load_state_dict(newstat)

    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr,weight_decay=config.wd)
    iter_num = len(train_loader.dataset) // config.batch_size
    aveGrad = 0
    net.train()

    for epoch in range(1, config.epoch + 1):
        r_sal_loss = 0
        net.zero_grad()
        for i, data_batch in enumerate(train_loader):
            rgb, t, label = data_batch['rgb_img'], data_batch['t_img'], data_batch['label']
            if (rgb.size(2) != label.size(2)) or (t.size(2) != label.size(2)) or (rgb.size(3) != label.size(3)) or (
                    t.size(3) != label.size(3)):
                print('IMAGE ERROR, PASSING```')
                continue
            if config.cuda:
                rgb, t, label = rgb.cuda(), t.cuda(), label.cuda()
            sal_pred = net(rgb, t)
            sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, label, reduction='sum')
            ##edge loss
            if config.edge_loss:
                edge_pred = lap(torch.sigmoid(sal_pred))
                edge_label = lap(label).detach()
                edge_loss = F.binary_cross_entropy(edge_pred, edge_label, reduction='sum')
                sal_loss = (0.7 * sal_loss_fuse + 0.3 * edge_loss) / (config.iter_size * config.batch_size)
            else:
                sal_loss = sal_loss_fuse / (config.iter_size * config.batch_size)

            r_sal_loss += sal_loss.data
            sal_loss.backward()
            aveGrad += 1
            if aveGrad % config.iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if i % (config.show_every // config.batch_size) == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %10.4f' % (
                    epoch, config.epoch, i, iter_num, r_sal_loss ))
                r_sal_loss = 0
        if epoch % config.epoch_save == 0:
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (config.save_folder,epoch))

        if epoch in config.lr_decay_epoch:
            optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr* 0.1,
                                  weight_decay=config.wd)
    torch.save(net.state_dict(), '%s/final.pth' % config.save_folder)

def test(test_loader,config):
    net = build_model(config)
    print('loading model from %s...' % config.model)
    if config.cuda:
        net = net.cuda()
        net.load_state_dict(torch.load(config.model))
    else:
        net.load_state_dict(torch.load(config.model, map_location='cpu'))
    net.eval()
    time_s = time.time()
    img_num = len(test_loader)
    for i, data_batch in enumerate(test_loader):
        rgb, t, name, im_size = data_batch['rgb'], data_batch['t'], data_batch['name'][0], np.asarray(
            data_batch['size'])
        with torch.no_grad():
            if config.cuda:
                rgb, t = rgb.cuda(), t.cuda()
            preds = net(rgb, t)
            pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
            multi_fuse = 255 * pred
            cv2.imwrite(os.path.join(config.test_fold, name[:-4] + '.png'), multi_fuse)
    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))



