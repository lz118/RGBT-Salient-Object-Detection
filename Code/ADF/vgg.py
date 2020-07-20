import torch
import torch.nn as nn

class MSA(nn.Module):
    def __init__(self,in_ch,useCA=True):
        super(MSA,self).__init__()
        self.useCA=useCA
        #for spatial mask   in: max_map  avg_map
        self.s_mask=nn.Sequential(
            nn.Conv2d(2, 1, 3, 1,1),
            nn.ReLU(),
            nn.Conv2d(1,1,3,1,1),
            nn.Sigmoid()
        )
        #for channel weight
        if self.useCA:
            self.avg_weight=nn.AdaptiveAvgPool2d(1)
            self.max_weight=nn.AdaptiveMaxPool2d(1)
            self.fus = nn.Sequential(
                nn.Conv2d(in_ch,in_ch//2,1,1,0),
                nn.ReLU(),
                nn.Conv2d(in_ch//2, in_ch, 1, 1, 0),
            )
            self.c_mask=nn.Sigmoid()

    def forward(self,x):
        if self.useCA:
            avg_map_c = self.avg_weight(x)
            max_map_c = self.max_weight(x)
            c_mask = self.c_mask(torch.add(self.fus(avg_map_c),self.fus(max_map_c)))
            x = torch.mul(x,c_mask)
        max_map_s = torch.unsqueeze(torch.max(x,1)[0],1)
        avg_map_s = torch.unsqueeze(torch.mean(x,1),1)
        map_s = torch.cat((max_map_s,avg_map_s),1)
        s_mask = self.s_mask(map_s)
        x = torch.mul(x,s_mask)
        return x

# vgg16
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    stage = 1
    for v in cfg:
        if v == 'M':
            stage += 1
            if stage == 6:
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            else:
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        else:
            if stage == 6:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

class a_vgg16(nn.Module):
    def __init__(self,config):
        super(a_vgg16, self).__init__()
        self.cfg =[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [3, 8, 15, 22, 29] # [ 8, 15, 22, 29]
        # 64:1 -->128:1/2 -->256:1/4 -->512 :1/8 --> 512:1/16 -->M-> 512,1/16
        self.base = nn.ModuleList(vgg(self.cfg, 3))

        #MSA
        self.useMSA=config.useMSA
        if self.useMSA:
            self.feature_channels=[64,128,256,512,512]
            msas=[]
            for fc in self.feature_channels:
                msas+=[MSA(fc,config.useCA)]
            self.msas=nn.ModuleList(msas)
        else:
            print('no MSA used!')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def load_pretrained_model(self, model):
        self.base.load_state_dict(model, strict=False)

    def forward(self, x):
        tmp_x = []
        index=0
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                if self.useMSA:
                    x=self.msas[index](x)
                    index+=1
                tmp_x.append(x)     #collect feature maps 1(64)  1/2(128)  1/4(256)  1/8(512)  1/16(512)
        return tmp_x

