import torch
from torch import nn
import torch.nn.functional as F
import vgg

class fusnet(nn.Module):
    def __init__(self):
        super(fusnet, self).__init__()
        self.extract=[]
        layers=[]
        self.extract = [4, 11, 18, 25]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                   nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),  # 128  1/2
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                   nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),  # 256  1/4
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                   nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),  # 512  1/8
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),  # 512  1/16
                   ]
        self.base=nn.ModuleList(layers)

        #PPM
        self.in_planes = 512
        self.out_planes = [512, 256, 128]
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(
                nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False),
                              nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)
        # 对四个子分支输出进行拼接  然后通过 3*3卷积 relu  降成512通道
        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False),
                                     nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

    def forward(self,rgb_f,t_f):
        x=torch.add(rgb_f[0],t_f[0])
        index=0
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)  # 收集每一尺度的输出特
                #融合分支下一卷积块的输入 上一块的输出加上 rgb  t对应输出
                index+=1
                x=torch.add(x,torch.add(rgb_f[index],t_f[index]))
        tmp_x.append(x)  # 分别�?1/2  1/4  1/8  1/16   �?融合rgb和t�?1/16

        #PPM forword  输出四个 GGF 分别 1/16  1/8   1/4  1/2
        xls = [tmp_x[-1]]  # 取融合模块最后一层的输出(融合了rgb t)     xls用于收集PPM四个分支的输�?
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](tmp_x[-1]), tmp_x[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))  # 得到PPM融合输出

        # 收集全局引导流信�? size为vgg对应层输出特�?
        infos = [xls]
        for k in range(len(self.infos)):
            infos.append(self.infos[k](
                F.interpolate(xls, tmp_x[len(tmp_x) - 3 - k].size()[2:], mode='bilinear', align_corners=True)))
        return tmp_x[:-1],infos

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)  #融合下一尺度的特征和全局引导�? 经过3*3卷积 就是原文中F的模�?

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))  #依次将不同池化尺度的特征相加
        resl = self.relu(resl)          #相加后relu激�?
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)     #AFM输出上采样至下一尺度特征大小
        resl = self.conv_sum(resl)          #3*3卷积重组特征 并把通道降成下一尺度特征通道�?
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)      #降成1*1通道

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(config_vgg):
    deep_pool_layers, score_layers = [], []

    for i in range(len(config_vgg['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config_vgg['deep_pool'][0][i], config_vgg['deep_pool'][1][i], config_vgg['deep_pool'][2][i], config_vgg['deep_pool'][3][i])]

    score_layers = ScoreLayer(config_vgg['score'])

    return deep_pool_layers, score_layers



class APMFnet(nn.Module):
    def __init__(self,config,deep_pool_layers, score_layers):
        super(APMFnet,self).__init__()
        self.config=config
        self.fusnet=fusnet()
        self.rgb_net= vgg.a_vgg16(config)
        self.t_net= vgg.a_vgg16(config)        #两个分支
        self.deep_pool = nn.ModuleList(deep_pool_layers)        #MSA + F
        self.score = score_layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,rgb,t):
        x_size = rgb.size()
        #先通过两个分支的输入计�?融合分支的输出fu_f  和全局引导流ggfs
        rgb_f= self.rgb_net(rgb)
        t_f= self.t_net(t)
        fu_f,ggfs= self.fusnet(rgb_f,t_f)
        fu_f = fu_f[::-1]       #列表倒置  [1,2,3,4]-->[4,3,2,1]   vgg:此处 �?(512 1/16) -->(512 1/8) -->(256 1/4)-->(128 1/2)
        feature_num=len(fu_f)

        #解码阶段应该�?将PPM输出的特征（1/16 512）与融合层最后一层特征（1/16 512）结合作为初始输�?
        merge=torch.add(fu_f[0],ggfs[0])  #512 1/16

        for k in range(feature_num-1):       #0,1 ,2
            merge = self.deep_pool[k](merge, fu_f[k+1], ggfs[k+1])  #输出依次�?�?12 1/8 �?:256  1/4  �?:128  1/2

        merge=self.deep_pool[-1](merge)  #最后一�?AFM模块 没有后续信息的融�?  128  1/2

        merge = self.score(merge, x_size)   #在上采样中用�?x_size[-2:]  -->(480,640)

        return merge

def build_model(config):
    return APMFnet(config,*extra_layer(config.config_vgg))
