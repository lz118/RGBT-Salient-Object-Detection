coding='utf-8'
import config
import os
import solver
from dataset import get_loader
import torch
import random
import numpy as np

if __name__ == '__main__':

    random.seed(1185)
    np.random.seed(1185)
    torch.manual_seed(1185)
    if config.cuda:
        torch.cuda.set_device(config.GPUid)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(1185)
        torch.cuda.manual_seed_all(1185)

    if config.mode == 'train':
        train_loader = get_loader(config)
        if not os.path.exists(config.save_folder):  os.mkdir(config.save_folder)
        train = solver.train(train_loader,config)
    elif config.mode == 'test':
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = solver.test(test_loader, config)
    else:
        print("ERROR MODE!")