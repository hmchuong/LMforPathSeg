import glob
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import PIL
import copy
from torchvision.transforms import transforms
import os
from PIL import ImageFile
import cv2
import os
import h5py
import pandas as pd
import torch.utils.data as data_utils
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from . import augmentation as psp_trsform
import albumentations as A
import albumentations.pytorch as AP

ImageFile.LOAD_TRUNCATED_IMAGES = True

from pyseg.utils.utils import init_log
import logging
logger =init_log('internal', logging.INFO)

class Camelyon16Dataset(data_utils.Dataset):

    def __init__(self, path=None, mode="train", transform=None, cfg=None):
        super().__init__()
        self.transform = transform
        self.return_image_rle = False
        self.config = {'STD_THRESHOLD': 10, 'MULTIPLIER_BIN': 4}
        self.cfg = cfg

        print('Loading Camelyon16 {} dataset...'.format(mode))

        # Open the files
        df = pd.read_csv(path + '/data.csv')

        if mode == 'train':
            df = df[df['filename_img'].str.count("^tumor_0(15|19)_.*") > 0]
            df = self.__filter_data(df, bin_counts=4, bin_ratio=[0, 1, 1, 1])
            # df = df.head(100)
            images = df['filename_img'].to_numpy()
            rles = df['filename_rle'].to_numpy()
            # images = df[df['filename_img'].str.count("^normal_00[1-8].*|^tumor_00[1-4].*") > 0][
            #     'filename_img'].to_numpy()
            # rles = df[df['filename_img'].str.count("^normal_00[1-8].*|^tumor_00[1-4].*") > 0]['filename_rle'].to_numpy()

        elif mode == 'val':
            df = df[df['std_img'] > self.config["STD_THRESHOLD"]]
            df = df[df['filename_img'].str.count("^tumor_034.*") > 0]
            df = self.__filter_data(df, bin_counts=4, bin_ratio=[0, 1, 1, 1])
            # df = df.head(100)
            # df = df.sample(frac=1).reset_index(drop=True).sample(n=5000)  # shuffle and then sample
            images = df['filename_img'].to_numpy()
            rles = df['filename_rle'].to_numpy()

        elif mode == 'test':
            df = df[df['std_img'] > self.config["STD_THRESHOLD"]]
            self.return_image_rle = True
            # df = df[df['filename_img'].str.count("^tumor_036.*|^tumor_034.*|^tumor_024.*") > 0]
            df = df[df['filename_img'].str.count("^tumor_006.*|^tumor_008.*|^tumor_020.*") > 0]
            df = df.sample(frac=1).reset_index(drop=True).sample(n=5000)  # shuffle and then sample
            self.test_df = df
            images = df['filename_img'].to_numpy()
            rles = df['filename_rle'].to_numpy()

        elif mode == 'plot':
            self.return_image_rle = True
            # self.test_df = df[df['filename_img'].str.count("^tumor_{}.*".format(str(lamel_idx).zfill(3))) > 0]
            lamel_idx = 19
            tum_num = "^tumor_{num}.*".format(num=str(lamel_idx).zfill(3))
            self.test_df = df[df['filename_img'].str.count(tum_num) > 0]
            self.test_df = self.test_df.sort_values("filename_img")
            self.test_df["x_index"] = self.test_df["x_index"].astype('str')
            self.test_df["y_index"] = self.test_df["y_index"].astype('str')
            self.test_df["x_index"] = self.test_df["x_index"].apply(lambda x: x[1:-1] if '[' in x else x)
            self.test_df["y_index"] = self.test_df["y_index"].apply(lambda x: x[1:-1] if '[' in x else x)
            self.test_df["x_index"] = self.test_df["x_index"].astype('int')
            self.test_df["y_index"] = self.test_df["y_index"].astype('int')
            images = self.test_df['filename_img'].to_numpy()
            rles = self.test_df['filename_rle'].to_numpy()
        # Read into numpy array

        self.X = images
        self.y = rles
        # self.path = path
        self.path = os.path.join(path, "lamels")

        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 50)

    def __filter_data(self, data, bin_counts, bin_ratio):
        """
        balancing data sampling with respect to masked area
        :param data: pandas dataframe
        :param patch_number: number of patches to be sampled
        :param bin_counts: number of bins
        :param bin_ratio: ratio of bins
        :return: balanced data as pandas dataframe
        """
        data = data[data['std_img'] > self.config['STD_THRESHOLD']].reset_index(drop=True)
        print('data.shape = ', data.shape)

        data['binned'] = np.round(data['ratio_masked_area'] * self.config['MULTIPLIER_BIN']).astype(int)
        max_bin = bin_counts - 1
        data['binned'] = data['binned'].apply(lambda x: max_bin if x >= max_bin else x)
        print('bin shape = ', data['binned'].value_counts())

        patch_number = data[data['binned'] == 0].shape[0] / 0.7

        data_balanced_list = []
        for bin in range(bin_counts):
            data_balanced_list.append(
                data[data['binned'] == bin].sample(int(patch_number * (bin_ratio[bin] / sum(bin_ratio))), replace=True))
            # print(len(data_balanced_list[-1]))
        data_balanced = pd.concat(data_balanced_list, axis=0)

        return data_balanced

    def __getitem__(self, idx):
        # sub_dir = self.X[idx][:9]
        path_dir = self.path #os.path.join(self.path, sub_dir)
        
        image = Image.open(path_dir + '/' + self.X[idx])
        label = 1 if len(pd.read_pickle(path_dir + '/' + self.y[idx])) else 0
        rle = pd.read_pickle(path_dir + '/' + self.y[idx])
        mask = rle2mask(rle, (1024, 1024))
        image = np.array(image)
        mask = np.array(mask)
        # start_time = time.time()
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        # logger.info(time.time() - start_time)
        image = image.squeeze(0)
        mask = mask.squeeze()
        # logger.info(mask.sum() / (mask.shape[0] * mask.shape[1]))
        if self.return_image_rle:
            return {"pos": (self.test_df.iloc[idx]["x_index"], self.test_df.iloc[idx]["y_index"]),
                    "image": np.array(image.getdata()).reshape(image.size[0], image.size[1], 3),
                    "transformed": transformed, "label": label, "rle": rle, "filename_img": self.X[idx]}
        
        return image, mask

    def __len__(self):
        return len(self.X)


def rle2mask(rle, rle_shape):
    split_rle = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (split_rle[0:][::2], split_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    image = np.zeros(rle_shape[0] * rle_shape[1], dtype=np.uint8)
    for start_point, end_point in zip(starts, ends):
        image[start_point:end_point] = 1
    image = image.reshape(rle_shape, order='F')  # Needed to align to RLE direction
    return image
    # return cv2.resize(image, mask_shape)


# def build_transfrom(cfg):
#     trs_form = []
#     mean, std, ignore_label = cfg['mean'], cfg['std'], cfg['ignore_label']
#     trs_form.append(psp_trsform.ToTensor())
#     trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
#     if cfg.get('resize', False):
#         trs_form.append(psp_trsform.Resize())
#     if cfg.get('rand_resize', False):
#         trs_form.append(psp_trsform.RandResize(cfg['rand_resize']))
#     if cfg.get('rand_rotation', False):
#         rand_rotation = cfg['rand_rotation']
#         trs_form.append(psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label))
#     if cfg.get('GaussianBlur', False) and cfg['GaussianBlur']:
#         trs_form.append(psp_trsform.RandomGaussianBlur())
#     if cfg.get('flip', False) and cfg.get('flip'):
#         trs_form.append(psp_trsform.RandomHorizontalFlip())
#     if cfg.get('crop', False):
#         crop_size, crop_type = cfg['crop']['size'], cfg['crop']['type']
#         trs_form.append(psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label))
#     return psp_trsform.Compose(trs_form)

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
    if cfg.get('rand_rotation', False):
        rand_rotation = cfg['rand_rotation']
        trns_form.append(A.Rotate(limit=rand_rotation))
    trns_form += [A.Normalize(mean=mean, std=std), AP.ToTensorV2()]
    return A.Compose(trns_form)


def build_camloader(split, all_cfg):
    cfg_dset = all_cfg['dataset']
    cfg_trainer = all_cfg['trainer']

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get('workers', 2)
    batch_size = cfg.get('batch_size', 1)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = Camelyon16Dataset(cfg['data_root'], split, trs_form, cfg)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    # build sampler
    if distributed:
        sample = DistributedSampler(dset)

        loader = DataLoader(dset, batch_size=batch_size if (split == 'train') else 1, num_workers=workers if (split == 'train') else 1,
                            sampler=sample, shuffle=False, pin_memory=False)
    else:
        
        loader = DataLoader(dset, batch_size=batch_size if (split == 'train') else 1, num_workers=workers if (split == 'train') else 1, shuffle=(split == 'train'), pin_memory=False)
    return loader

    