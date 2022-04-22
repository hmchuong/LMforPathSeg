import imp
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..base import  conv1x1, conv3x3
from .contrast import dec_contrast

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1).apply(init_weight)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x

class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x  = F.relu(self.bn1(inputs))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.relu(self.bn2(x)))
        x  = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x

class dec_unet_contrast(dec_contrast):
   
    def __init__(self, num_classes=2, inner_planes=320, temperature=0.2, queue_len=2975, **kwargs):
        super(dec_unet_contrast, self).__init__(inner_planes, num_classes, temperature, queue_len, **kwargs)

        # bottleneck
        self.center  = CenterBlock(2048, 512)

        # decoder
        self.decoder4 = DecodeBlock(512 + 2048, 64, upsample=True)
        self.decoder3 = DecodeBlock(64 + 1024, 64, upsample=True)
        self.decoder2 = DecodeBlock(64 + 512, 64, upsample=True)
        self.decoder1 = DecodeBlock(64 + 256, 64, upsample=True)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)

        # upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64, num_classes).apply(init_weight)
        )

    def forward(self, x1, x2, x3, x4):
     
        y5 = self.center(x4) #->(*,320,h/32,w/32)

        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)

        logits = self.final_conv(hypercol)
        if not self.training:
            return logits
        return super().forward(hypercol, logits)