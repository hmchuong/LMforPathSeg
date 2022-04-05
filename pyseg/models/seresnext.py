import torch.nn as nn
import pretrainedmodels

class SEResNextEncoder(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super().__init__()
        model_name = "se_resnext101_32x4d"
        seresnext101 = pretrainedmodels.__dict__[model_name](pretrained=pretrained)

        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool, #->(*,64,h/4,w/4)
            seresnext101.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4 #->(*,2048,h/32,w/32)
    
    def get_outplanes(self):
        return 2048

    def forward(self, inputs):

        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2)
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)

        return x1, x2, x3, x4

def seresnext101(pretrained='imagenet', **kwargs):
    model = SEResNextEncoder(pretrained)
    return model