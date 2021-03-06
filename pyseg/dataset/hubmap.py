
import numpy as np
from PIL import Image

import copy
from torchvision.transforms import transforms
import os
from PIL import ImageFile
import cv2
import os

import pandas as pd
import torch.utils.data as data_utils

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import albumentations as A
import albumentations.pytorch as AP

ImageFile.LOAD_TRUNCATED_IMAGES = True

# from pyseg.utils.utils import init_log
# import logging
# logger =init_log('internal', logging.INFO)

IDS = ['0486052bb',
 '095bf7a1f',
 '1e2425f28',
 '26dc41664',
 '2f6ecfcdf',
 '4ef6695ce',
 '54f2eec69',
 '8242609fa',
 'aaa6a05cc',
 'afa5e8098',
 'b2dc8411c',
 'b9a3865fc',
 'c68fe75ea',
 'cb2d976f4',
 'e79de561c']

class HubmapDataset(data_utils.Dataset):

    def __init__(self, path=None, mode="train", transform=None, cfg=None, return_name=False, whole_images=None,
                 whole_rles=None,
                 custom_idx=None):
        super().__init__()
        self.transform = transform
        self.cfg = cfg

        print('Loading Hubmap {} dataset...'.format(mode))

        # Open the files
        df = pd.read_csv(path + '/data.csv')
        ratio_threshold = 0

        if mode == "hard_mining":
            images = whole_images
            rles = whole_rles
            if custom_idx is not None:
                images = images[custom_idx]
                rles = rles[custom_idx]
        if mode == 'train':
            filter_str = "^./images/(" + "|".join(IDS[:10]) + ")_*"
            df = df[df['image'].str.count(filter_str) > 0]
            # df = df[df['ratio'] > ratio_threshold]
            images = df['image'].to_numpy()
            rles = df['mask'].to_numpy()

        elif mode == 'val':
            filter_str = "^./images/(" + "|".join(IDS[10:11]) + ")_*"
            df = df[df['image'].str.count(filter_str) > 0]
            images = df['image'].to_numpy()
            rles = df['mask'].to_numpy()

        elif mode == 'test':
            filter_str = "^./images/(" + "|".join(IDS[11:]) + ")_*"
            df = df[df['image'].str.count(filter_str) > 0]
            images = df['image'].to_numpy()
            rles = df['mask'].to_numpy()

        # Read into numpy array

        self.X = images
        self.y = rles
        self.path = path
        self.return_name = return_name
        self.whole_images = images
        self.whole_rles = rles

        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 50)

    def __getitem__(self, idx):

        path_dir = self.path
        
        image = Image.open(path_dir + '/' + self.X[idx])
        mask = cv2.imread(path_dir + '/' + self.y[idx], cv2.IMREAD_GRAYSCALE)
        image = np.array(image)
        mask = (mask > 128).astype(int)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = image.squeeze(0)
            mask = mask.squeeze()
        # logger.info(mask.sum() / (mask.shape[0] * mask.shape[1]))
        if self.return_name:
            return image, mask, self.X[idx].split('/')[-1]
        return image, mask, idx

    def __len__(self):
        return len(self.X)

def build_transfrom(cfg):
    trns_form = []
    mean, std, ignore_label = cfg['mean'], cfg['std'], cfg['ignore_label']
    if cfg.get('resize', False):
        width, height = cfg['resize']
        trns_form.append(A.Resize(height=height, width=width))
    if cfg.get('rand_resize_crop', False):
        scale = cfg['rand_resize_crop']['scale']
        w, h = cfg['rand_resize_crop']['size']
        trns_form.append(A.RandomResizedCrop(height=h, width=w, scale=tuple(scale)))
    if cfg.get('flip', False) and cfg.get('flip'):
        trns_form.append(A.HorizontalFlip(p=0.5))
        trns_form.append(A.VerticalFlip(p=0.5))
    if cfg.get('rand_rotation', False):
        rand_rotation = cfg['rand_rotation']
        trns_form.append(A.Rotate(limit=rand_rotation))
    trns_form += [A.Normalize(mean=mean, std=std), AP.ToTensorV2()]
    return A.Compose(trns_form)


def build_hubmaploader(split, all_cfg):
    cfg_dset = all_cfg['dataset']
    cfg_trainer = all_cfg['trainer']

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get('workers', 2)
    batch_size = cfg.get('batch_size', 1)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = HubmapDataset(cfg['data_root'], split, trs_form, cfg, return_name=(split=='test'))

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    # build sampler
    if distributed:
        sample = DistributedSampler(dset)

        loader = DataLoader(dset, batch_size=batch_size if (split == 'train') else 1, num_workers=workers if (split == 'train') else 1,
                            sampler=sample, shuffle=False)
    else:
        
        loader = DataLoader(dset, batch_size=batch_size if (split == 'train') else 1, num_workers=workers if (split == 'train') else 1, shuffle=(split == 'train'))
    return loader


if __name__ == "__main__":
    data_root = "/fs/class-projects/spring2022/cmsc828l/c828lg001/hubmap_patches"
    split = "train"

    dset = HubmapDataset(data_root, split, None, None, return_name=False)
    areas = []
    from skimage import measure
    for i, (_, label) in enumerate(dset):
        regions = measure.label(label)
        H, W = label.shape
        for label_num in np.unique(regions):
            if label_num == 0: continue
            area = (regions == label_num).sum()
            # area /= ( H * W)
            areas += [area]
        print("Done", i)
    import pdb; pdb.set_trace() 



    