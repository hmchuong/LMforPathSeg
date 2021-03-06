from glob import glob
import torch
import gc
import torch.nn as nn
from torch.nn import functional as F
from skimage import measure
import numpy as np
# from tensorboardX import SummaryWriter
from ..base import  ASPP, get_syncbn

class dec_contrast(nn.Module):
    def __init__(self, inner_planes, num_classes=19, temperature=0.2, queue_len=2975, contrast_type='default', region_min=128, use_certainty=False, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.inner_planes = inner_planes
        self.contrast_type = contrast_type
        self.region_min = region_min
        self.use_certainty = use_certainty
        for i in range(num_classes):
            self.register_buffer("queue"+str(i),torch.randn(inner_planes, self.queue_len))
            self.register_buffer("ptr"+str(i),torch.zeros(1,dtype=torch.long))
            exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0)')
        
        # self.writer = SummaryWriter(comment='hubmap_embedding_default_0418')
        # self.global_step = 0
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self,keys,vals,cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        batch_size = bs
        ptr = int(eval("self.ptr"+str(cat)))
        eval("self.queue"+str(cat))[:,ptr] = keys
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

        # Store to writer
        # with open("embedding_noHM.tsv", "a") as f:
        #     f.write('\t'.join([str(x) for x in keys.cpu().numpy().tolist()]) + "\n")
        # with open("embedding_noHM_label.txt", "a") as f:
        #     f.write(str(cat) + "\n")

        del keys, ptr
        gc.collect()

    @torch.no_grad()
    def _dequeue_and_enqueue_unconnected(self, keys, vals, cat):
        if cat not in vals:
            return
        keys = keys[vals == cat]
        batch_size = keys.shape[0]
        keys = keys.permute(1, 0)
        ptr = int(eval("self.ptr" + str(cat)))

        # label_f = open("embedding_uncon128_label.txt", "a")
        # with open("embedding_uncon128.tsv", "a") as f:
        #     for i in range(keys.shape[1]):
        #         f.write('\t'.join([str(x) for x in keys[:, i].cpu().numpy().tolist()]) + "\n")
        #         label_f.write(str(cat) + "\n")
        # label_f.close()

        if ptr + batch_size > self.queue_len:
            # import pdb; pdb.set_trace()
            batch_size = self.queue_len - ptr
            keys = keys[:, :batch_size]
        eval("self.queue"+str(cat))[:, ptr: ptr + batch_size] = keys.detach()
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr"+str(cat))[0] = ptr

        

        del keys, ptr
        gc.collect()
        
    def construct_region(self, fea, pred):
        
        bs = fea.shape[0]
        certainty = pred.softmax(1)
        certainty = (certainty[:, 0] - certainty[:, 1]).abs()
        certainty = certainty.squeeze().view(bs, -1)
        pred = pred.max(1)[1].squeeze().view(bs, -1)  
       
        val = torch.unique(pred)
        fea=fea.squeeze()
        fea = fea.view(bs, self.inner_planes,-1).permute(1,0,2)   
        if not self.use_certainty:
            new_fea = fea[:,pred==val[0]].mean(1)
        else:
            new_fea = (fea[:,pred==val[0]] * certainty[pred == val[0]]).sum(1) 
            new_fea = new_fea/ (certainty[pred == val[0]]).sum()
        new_fea = new_fea.unsqueeze(0)
        for i in val[1:]:
            if(i<19):
                if not self.use_certainty:
                    class_fea = fea[:,pred==i].mean(1)  
                else:
                    class_fea = (fea[:,pred==i] * certainty[pred == i]).sum(1) 
                    class_fea = class_fea/ (certainty[pred == i]).sum()
                class_fea = class_fea.unsqueeze(0)
                new_fea = torch.cat((new_fea,class_fea),dim=0)
        # import pdb; pdb.set_trace()
        val = torch.tensor([i for i in val if i<19])
        return new_fea, val.cuda()

    def construct_unconnected_region(self, fea, pred, threshold=128):
        
        bs = fea.shape[0]
        certainty = pred.softmax(1)
        certainty = (certainty[:, 0] - certainty[:, 1]).abs()
        pred = pred.max(1)[1].squeeze()  
        
        comp = measure.label(pred.cpu().numpy())
        val = []
        new_fea = []
        threshold = self.region_min if self.region_min > 1.0 else (pred.shape[1] * pred.shape[2]) * self.region_min
        for i in range(bs):
            for label in np.unique(comp[i]):
                mask = comp[i] == label
                if mask.sum() < threshold:
                    continue
                mask = torch.from_numpy(mask).cuda()
                if not self.use_certainty:
                    new_fea += [fea[i, :,mask].mean(1)]
                else:
                    class_fea = (fea[i, :, mask] * certainty[i, mask]).sum(1) 
                    class_fea = class_fea/ (certainty[i, mask]).sum()
                    new_fea += [class_fea]
                val += [1 if label != 0 else 0]

        new_fea = torch.stack(new_fea)
        val = torch.tensor(val)

        return new_fea, val.cuda()

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)
        logits = torch.cat((l_pos,l_neg),dim=1)
        logits /= self.temperature
        labels = torch.zeros((N,),dtype=torch.long).cuda()
        return self.criterion(logits,labels)
    
    def _forward_unconnected(self, fea, res):
        keys, vals = self.construct_unconnected_region(fea, res)  #keys: N,256   vals: N,  N is the category number in this batch
        # import pdb; pdb.set_trace()
        keys = nn.functional.normalize(keys,dim=1)
        contrast_loss = 0

        for key, val in zip(keys, vals):
            query = key
            l_pos = query.unsqueeze(1)*eval("self.queue"+str(val.item())).clone().detach()  #256, N1
            all_ind = [m for m in range(self.num_classes)]
            l_neg = 0
            tmp = all_ind.copy()
            tmp.remove(val.item())
            for cls_ind2 in tmp:
                l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
            contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
        contrast_loss = contrast_loss / keys.shape[0]
        # import pdb; pdb.set_trace()
        for i in range(self.num_classes):
            self._dequeue_and_enqueue_unconnected(keys, vals, i)
        return res, contrast_loss

    def _forward(self, fea, res):
        # return res, 1.0
        bs = res.shape[0]
        # keys, vals = self.construct_unconnected_region(fea, res)  #keys: N,256   vals: N,  N is the category number in this batch
        keys, vals = self.construct_region(fea, res)
        keys = nn.functional.normalize(keys,dim=1)
        contrast_loss = 0
        # import pdb; pdb.set_trace()
        for cls_ind in range(self.num_classes):
            if cls_ind in vals:
                query = keys[list(vals).index(cls_ind)]   #256,
                l_pos = query.unsqueeze(1)*eval("self.queue"+str(cls_ind)).clone().detach()  #256, N1
                all_ind = [m for m in range(self.num_classes)]
                l_neg = 0
                tmp = all_ind.copy()
                tmp.remove(cls_ind)
                for cls_ind2 in tmp:
                    l_neg += query.unsqueeze(1)*eval("self.queue"+str(cls_ind2)).clone().detach()
                contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
                del l_pos, l_neg, query, tmp, all_ind
                gc.collect()
            else:
                continue
        for i in range(self.num_classes):
            self._dequeue_and_enqueue(keys,vals,i, bs)
        del keys, vals
        gc.collect()
        torch.cuda.empty_cache()
        return res, contrast_loss
    
    def forward(self, fea, res):
        if self.contrast_type == 'default':
            return self._forward(fea, res)
        elif self.contrast_type == 'unconnected':
            return self._forward_unconnected(fea, res)

class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res

# Start: new BCE loss update
class Aux_Classification_Module(nn.Module):
    def __init__(self, in_planes, num_classes=1, sync_bn=False):
        super(Aux_Classification_Module, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
            nn.Linear(in_planes, 512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        res = self.pool(x)
        # import ipdb
        # ipdb.set_trace()
        res = res.view(res.size()[0], -1)
        res = self.fc1(res)
        return res.flatten()
# End: new BCE loss update
