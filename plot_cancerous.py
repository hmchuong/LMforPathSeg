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


# def apply_color_overlay(image, intensity=0.3, red=200, green=66, blue=0):
#     image = image.astype('uint8')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
#     h, w, c = image.shape
#     color_bgra = (blue, green, red, 1)
#     overlay = np.full((h, w, c), color_bgra, dtype='uint8')
#     cv2.addWeighted(overlay, intensity, image, 1.0, 0, image)
#     return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

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


class swav_light():
    def __init__(self):
        self.arch = "resnet50"
        self.global_pooling = True
        self.use_bn = False
        self.pretrained = r"./experiments/2/checkpoint.pth.tar"
        self.device = torch.device('cuda:1')
        self.model = resnet_models.__dict__[self.arch](output_dim=0, eval_mode=True).cuda(self.device)
        model_state_dict = torch.load(self.pretrained, map_location=self.device)["state_dict"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        self.model.load_state_dict(model_state_dict, strict=False)

        self.linear_classifier = RegLog(2, self.arch, self.global_pooling, self.use_bn).cuda(self.device)  ## 1000 ==> 2
        head_state_dict = \
            torch.load('/home/shahidzadeh/CENNALAB_AI/rep_learning/swav/experiments/2-head/checkpoint.best.pth.tar',
                       map_location=self.device)[
                "state_dict"]
        head_state_dict = {k.replace("module.", ""): v for k, v in head_state_dict.items()}
        self.linear_classifier.load_state_dict(head_state_dict)

        self.model.eval()
        self.linear_classifier.eval()

    def predict(self, batch):
        output = self.model(batch)
        output = self.linear_classifier(output)
        return output


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

    def predict(self, images):
        with torch.no_grad():
            preds = self.model(images)
        output = preds[0]
        output = output.data.max(1)[1].cpu()
        return output


data_path = "/fs/class-projects/spring2022/cmsc828l/c828lg001/camelyon16/"
model = RegCon()
cfg = copy.deepcopy(model.cfg["dataset"])
cfg.update(cfg.get("test", {}))
trns = build_transfrom(cfg, test_mode=True)
for lamel in range(20, 70):
    try:
        #     if lamel in [14, 15, 17, 19, 23, 25, 28, 30, 31, 34, 36, 39, 55, 61]:
        #         continue

        print("plotting", lamel, "...")
        test_dataset = Camelyon16Dataset(path=data_path, mode="plot", transform=trns, cfg=model.cfg, lamel_idx=lamel)
        test_data_df = test_dataset.test_df


        batch_size = 20
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
            pred = model.predict(images)
            # temperature = 20
            # cancer_probs = torch.nn.functional.softmax(output / temperature, dim=1)
            # pred = torch.argmax(cancer_probs, dim=1)
            # if not model:
            #     pred = np.zeros(batch_size)
            true_pred += torch.sum(torch.eq(pred, batch["mask"]))
            for i in range(batch_size):
                # i, j = np.unravel_index(test_data_df["filename_img"].split("_")[-1].split(".")[0],
                #                         shape=(row_count, col_count))
                # try:
                #     cernainty = max(cancer_probs[i]).item() / 5
                cernainty = 0.4
                if i == len(batch["pos"][0]):
                    break
                # start = time.time()
                c, r = batch["pos"][0][i].item(), batch["pos"][1][i].item()
                patch = batch["thumbnail"][i].numpy()

                if batch["label"][i] == 1:
                    patch = apply_color_overlay(patch, batch["mask"][i].numpy(), intensity=cernainty, red=200)
                if pred[i].sum() > 0:
                    patch = apply_color_overlay(patch, pred[i].numpy(), intensity=cernainty, green=200, red=50)
                # print("2", time.time() - start)

                whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size, :] = patch
                # print("3", time.time() - start)

                # if pred[i] and batch["label"][i]:
                #     masked = apply_color_overlay(batch["image"][i].numpy(), batch["mask"][i],
                #                                  intensity=cernainty, green=200, red=0)
                #     whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                #     :] = masked
                # elif batch["label"][i]:
                #     masked = apply_color_overlay(batch["image"][i].numpy(),
                #                                  intensity=cernainty, red=200)
                #     whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                #     :] = masked
                # elif pred[i]:
                #     masked = apply_color_overlay(batch["image"][i].numpy(),
                #                                  intensity=cernainty, green=150, red=150)
                #     whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                #     :] = masked
                # else:
                #     whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size, :] = \
                #         batch["image"][i].numpy()

                #     if batch["label"][i] == 1:
                #         cv2.imwrite("cancerous_red.jpg", masked)
                #         print(cernainty)
                #         print(pred[i])
                #         print(batch["label"][i])
                #         print(20*"---")
                sys.stdout.flush()
                sys.stdout.write("row:{r}, col:{c}\r".format(r=r, c=c))
                # whole_slide_image[128 * r:128 * (r + 1), c * 128:(c + 1) * 128, :] = masked
                # whole_slide_image_normal[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                # :] = \
                #     batch["image"][i].numpy()

                # except:
                # print('err')
                # break
        save_dir = "./slides/tumor__{}.jpg".format(lamel)
        cv2.imwrite(save_dir, whole_slide_image)
        # cv2.imwrite("whole_slide_image_normal.jpg", whole_slide_image_normal)
        # df = df[df['std_img'] <= 10]
        # for row in df.iterrows():
        #     image_num = row["filename_img"].split("_")[-1].split(".")[0]
        #     i, j = np.unravel_index(image_num, shape=(row_count, col_count))
        #     whole_slide_image[i, j] = cv2.imread(os.path.join(data_path, row["filename"]))
    except:
        print(lamel, "error")
        continue
