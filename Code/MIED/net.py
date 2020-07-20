import torch
from torch import nn
import torch.nn.functional as F
import vgg

def convblock(in_,out_,ks,st,pad):
    return nn.Sequential(
        nn.Conv2d(in_,out_,ks,st,pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class GFB(nn.Module):
    def __init__(self,in_1,in_2):
        super(GFB, self).__init__()
        self.ca1 = CA(2*in_1)
        self.conv1 = convblock(2*in_1,128, 3, 1, 1)
        self.conv_globalinfo = convblock(512,128,3, 1, 1)
        self.ca2 = CA(in_2)
        self.conv_curfeat =convblock(in_2,128,3,1,1)
        self.conv_out= convblock(128,in_2,3,1,1)

    def forward(self, pre1,pre2,cur,global_info):
        cur_size = cur.size()[2:]
        pre = self.ca1(torch.cat((pre1,pre2),1))
        pre =self.conv1(F.interpolate(pre,cur_size,mode='bilinear',align_corners=True))

        global_info = self.conv_globalinfo(F.interpolate(global_info,cur_size,mode='bilinear',align_corners=True))
        cur_feat =self.conv_curfeat(self.ca2(cur))
        fus = pre + cur_feat + global_info
        return self.conv_out(fus)

        
class GlobalInfo(nn.Module):
    def __init__(self):
        super(GlobalInfo, self).__init__()
        self.ca = CA(1024)
        self.de_chan = convblock(1024,256,3,1,1)
        
        self.b0 = nn.Sequential(
            nn.AdaptiveMaxPool2d(13),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )

        self.b1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(9),
            nn.Conv2d(256,128,1,1,0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(5),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(256, 128, 1, 1, 0,bias=False),
            nn.ReLU(inplace=True)
        )
        self.fus = convblock(768,512,1,1,0)
        
    def forward(self, rgb,t):
        x_size=rgb.size()[2:]
        x=self.ca(torch.cat((rgb,t),1))
        x=self.de_chan(x)
        b0 = F.interpolate(self.b0(x),x_size,mode='bilinear',align_corners=True)
        b1 = F.interpolate(self.b1(x),x_size,mode='bilinear',align_corners=True)
        b2 = F.interpolate(self.b2(x),x_size,mode='bilinear',align_corners=True)
        b3 = F.interpolate(self.b3(x),x_size,mode='bilinear',align_corners=True)
        out = self.fus(torch.cat((b0,b1,b2,b3,x),1))
        return out
        
class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class FinalScore(nn.Module):
    def __init__(self):
        super(FinalScore, self).__init__()
        self.ca =CA(256)
        self.score = nn.Conv2d(256, 1, 1, 1, 0)
    def forward(self,f1,f2,xsize):
        f1 = torch.cat((f1,f2),1)
        f1 = self.ca(f1)
        score = F.interpolate(self.score(f1), xsize, mode='bilinear', align_corners=True)
        return score

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.global_info =GlobalInfo()
        self.score_global = nn.Conv2d(512, 1, 1, 1, 0)

        self.gfb4_1 = GFB(512,512)
        self.gfb3_1= GFB(512,256)
        self.gfb2_1= GFB(256,128)

        self.gfb4_2 = GFB(512, 512) #1/8
        self.gfb3_2 = GFB(512, 256)#1/4
        self.gfb2_2 = GFB(256, 128)#1/2

        self.score_1=nn.Conv2d(128, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)

        self.refine =FinalScore()


    def forward(self,rgb,t):
        xsize=rgb[0].size()[2:]
        global_info =self.global_info(rgb[4],t[4]) # 512 1/16
        d1=self.gfb4_1(global_info,t[4],rgb[3],global_info)
        d2=self.gfb4_2(global_info, rgb[4], t[3], global_info)
        #print(d1.shape,d2.shape)
        d3= self.gfb3_1(d1, d2,rgb[2],global_info)
        d4 = self.gfb3_2(d2, d1, t[2], global_info)
        d5 = self.gfb2_1(d3, d4, rgb[1], global_info)
        d6 = self.gfb2_2(d4, d3, t[1], global_info) #1/2 128

        score_global = self.score_global(global_info)

        score1=self.score_1(F.interpolate(d5,xsize,mode='bilinear',align_corners=True))
        score2 = self.score_2(F.interpolate(d6, xsize, mode='bilinear', align_corners=True))
        score =self.refine(d5,d6,xsize)
        return score,score1,score2,score_global

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet,self).__init__()
        self.rgb_net= vgg.a_vgg16()
        self.t_net= vgg.a_vgg16()        #两个分支
        self.decoder=Decoder()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,rgb,t):
        rgb_f= self.rgb_net(rgb)
        t_f= self.t_net(t)
        score,score1,score2,score_g =self.decoder(rgb_f,t_f)
        return score,score1,score2,score_g

    def load_pretrained_model(self):
        st=torch.load("vgg16.pth")
        st2={}
        for key in st.keys():
            st2['base.'+key]=st[key]
        self.rgb_net.load_state_dict(st2)
        self.t_net.load_state_dict(st2)
        print('loading pretrained model success!')


if __name__=="__main__":
    a=torch.rand(1,3,400,400)
    b=a
    net=Mnet()
    c,c1,c2=net(a,b)
    print(c.shape)
    print(c1.shape)
    print(c2.shape)