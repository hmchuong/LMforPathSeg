import torch
import torch.nn as nn
from torch.nn import functional as F
from ..base import  ASPP, get_syncbn
from .contrast import dec_contrast

class dec_deeplabv3_contrast(dec_contrast):
   
    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), temperature=0.2, queue_len=2975):
        super(dec_deeplabv3_contrast, self).__init__(inner_planes, num_classes, temperature, queue_len)

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))
        self.final =  nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
     
        aspp_out = self.aspp(x)
        fea = self.head(aspp_out)
        res = self.final(fea)
        if not self.training:
            return res
        return super().forward(fea, res)


