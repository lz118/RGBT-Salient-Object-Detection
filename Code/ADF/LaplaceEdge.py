import torch.nn as nn
import torch
import numpy as np
#laplace operator layer for edge extraction

class Laplace_edge_layer(nn.Module):
    def __init__(self):
        super(Laplace_edge_layer,self).__init__()
        self.laplace=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.weight_init()
    def weight_init(self):
        self.laplace.weight.data=torch.tensor(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])[np.newaxis,np.newaxis,...]).float()
    def forward(self, x):
        return torch.abs(self.laplace(x)/9)

# class Laplace_edge_layer(nn.Module):
#     def __init__(self):
#         super(Laplace_edge_layer,self).__init__()
#         self.laplace=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
#         self.act=nn.Tanh()
#         self.weight_init()
#     def weight_init(self):
#         self.laplace.weight.data=torch.tensor(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])[np.newaxis,np.newaxis,...]).float()
#     def forward(self, x):
#         return self.act(torch.abs(self.laplace(x)))
#
# def iou_edge_loss(edge,label):
#     edge, label = torch.squeeze(edge,1), torch.squeeze(label,1)
#     inter = torch.sum(torch.mean(torch.mul(edge,label),0))
#     union = torch.add(torch.sum(torch.mean(torch.mul(edge,edge),0)),torch.sum(torch.mean(torch.mul(label,label),0)))
#     loss = torch.add(1, -(2 * inter) / union)
#     return loss


