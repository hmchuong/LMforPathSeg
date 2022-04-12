import cv2
import os
import sys
import copy
import argparse
import yaml
import os.path as osp
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pyseg.models.model_helper import ModelBuilder
from pyseg.dataset.camelyon16 import Camelyon16Dataset, build_transfrom
from pyseg.utils.loss_helper import get_criterion
from pyseg.utils.lr_helper import get_scheduler, get_optimizer

from pyseg.utils.utils import AverageMeter, intersectionAndUnion, init_log, load_trained_model
from pyseg.utils.utils import set_random_seed, get_world_size, get_rank, is_distributed
from pyseg.dataset.builder import get_loader
from sklearn.manifold import TSNE


def apply_color_overlay(image, mask=None, intensity=0.3, red=200, green=66, blue=0):
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, c = image.shape
    color_bgra = (blue, green, red, 1)
    if mask is not None:
        mask = mask.astype('uint8')
        mask = cv2.resize(mask, (h, w))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
        overlay = mask * color_bgra
    else:
        overlay = np.full((h, w, c), color_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, image, 1.0, 0, image, dtype=cv2.CV_8U)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


class RegCon():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch Semantic Segmentation")
        parser.add_argument("--config", type=str,
                            default="/fs/classhomes/spring2022/cmsc828l/c828l050/RegionContrast-Med/experiments/camelyon/config_contrast.yaml")
        args = parser.parse_args()
        self.cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

        cudnn.enabled = True
        cudnn.benchmark = True
        self.model = ModelBuilder(self.cfg['net'])
        device = torch.device("cuda")
        self.model.to(device)
        state_dict = torch.load(osp.join("experiments", "camelyon", self.cfg['test']['model']), map_location='cpu')[
            'model_state']
        load_trained_model(self.model, state_dict)
        self.model.eval()

    def predict(self, images, logits=True):
        with torch.no_grad():
            preds = self.model(images)
        output = preds[0]
        if logits:
            return output.data.max(1)[1].cpu()
        else:
            return output[:, 1, :, :].cpu()


data_path = "/fs/class-projects/spring2022/cmsc828l/c828lg001/camelyon16/"
model = RegCon()
cfg = copy.deepcopy(model.cfg["dataset"])
cfg.update(cfg.get("test", {}))
trns = build_transfrom(cfg, test_mode=True)
batch_size = 20
for lamel in range(14, 70):
    try:

        print("plotting", lamel, "...")
        test_dataset = Camelyon16Dataset(path=data_path, mode="plot", transform=trns, cfg=model.cfg, lamel_idx=lamel)
        test_data_df = test_dataset.test_df

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )
        if len(test_loader) == 0:
            continue
        col_count = max(test_data_df["x_index"]) + 3

        row_count = max(test_data_df["y_index"]) + 3
        print("with shape:", row_count, col_count)
        tile_size = 100
        whole_slide_image_normal = np.zeros((tile_size * row_count, tile_size * col_count, 3))
        whole_slide_image = np.zeros((tile_size * row_count, tile_size * col_count, 3))
        print(whole_slide_image.shape)
        true_pred = 0
        for batch in test_loader:
            images = batch["image"].cuda()
            pred = model.predict(images, logits=False)
            # if not model:
            #     pred = np.zeros(batch_size)
            # true_pred += torch.sum(torch.eq(pred, batch["mask"]))
            for i in range(batch_size):
                cernainty = 0.8
                if i == len(batch["pos"][0]):
                    break
                c, r = batch["pos"][0][i].item(), batch["pos"][1][i].item()
                patch = batch["thumbnail"][i].numpy()
                if batch["label"][i] == 1:
                    patch = apply_color_overlay(patch, batch["mask"][i].numpy(), intensity=cernainty, green=200, red=200)
                # if pred[i].sum() > 0:
                #     patch = apply_color_overlay(patch, pred[i].numpy(), intensity=cernainty, green=200, red=50)

                whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size, :] = patch
                sys.stdout.flush()
                sys.stdout.write("row:{r}, col:{c}\r".format(r=r, c=c))
        os.makedirs("./slides_tumor/", exist_ok=True)

        save_dir = "./slides_tumor/tumor_{}.jpg".format(lamel)
        cv2.imwrite(save_dir, whole_slide_image)
    except:
        print(lamel, "error")
        continue
