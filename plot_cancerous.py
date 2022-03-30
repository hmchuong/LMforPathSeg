import cv2
import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pyseg.dataset.camelyon16 import Camelyon16Dataset
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


def apply_color_overlay(image, intensity=0.3, red=200, green=66, blue=0):
    image = image.astype('uint8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    h, w, c = image.shape
    color_bgra = (blue, green, red, 1)
    overlay = np.full((h, w, c), color_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, image, 1.0, 0, image)
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


data_path = "/fs/class-projects/spring2022/cmsc828l/c828lg001/camelyon16/"

for lamel in range(14, 70):
    # try:
    #     if lamel in [14, 15, 17, 19, 23, 25, 28, 30, 31, 34, 36, 39, 55, 61]:
    #         continue

        print("plotting", lamel, "...")
        test_dataset = Camelyon16Dataset(path=data_path, mode="plot", lamel_idx=lamel)
        batch_size = 8
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            # collate_fn=lambda x: x['transformed'].to(torch.device('cuda:1'))
        )

        test_data_df = test_dataset.test_df
        if len(test_loader) == 0:
            continue
        col_count = max(test_data_df["x_index"]) + 3
        row_count = max(test_data_df["y_index"]) + 3
        # model = swav_light()
        print("with shape:", row_count, col_count)
        tile_size = 100
        whole_slide_image_normal = np.zeros((tile_size * row_count, tile_size * col_count, 3))
        whole_slide_image = np.zeros((tile_size * row_count, tile_size * col_count, 3))
        print(whole_slide_image.shape)
        true_pred = 0
        for batch in test_loader:
            # inp = batch["transformed"].to(torch.device('cuda:1'))
            # output = model.predict(inp)
            # temperature = 20
            # cancer_probs = torch.nn.functional.softmax(output / temperature, dim=1)
            # pred = torch.argmax(cancer_probs, dim=1)
            pred = np.zeros(batch_size)
            # true_pred += torch.sum(torch.eq(pred, batch["label"].cuda(torch.device("cuda:1"))))
            for i in range(batch_size):
                # i, j = np.unravel_index(test_data_df["filename_img"].split("_")[-1].split(".")[0],
                #                         shape=(row_count, col_count))
                # try:
                #     cernainty = max(cancer_probs[i]).item() / 5
                cernainty = 0.4
                if i == len(batch["pos"][0]):
                    break
                c, r = batch["pos"][0][i].item(), batch["pos"][1][i].item()
                # if batch["label"][i] == 1:
                #     cv2.imwrite("cancerous.jpg", batch["image"][i].numpy())
                if pred[i] and batch["label"][i]:
                    masked = apply_color_overlay(batch["image"][i].numpy(),
                                                 intensity=cernainty, green=200, red=0)
                    whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                    :] = masked
                elif batch["label"][i]:
                    masked = apply_color_overlay(batch["image"][i].numpy(),
                                                 intensity=cernainty)
                    whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                    :] = masked
                elif pred[i]:
                    masked = apply_color_overlay(batch["image"][i].numpy(),
                                                 intensity=cernainty, green=150, red=150)
                    whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                    :] = masked
                else:
                    whole_slide_image[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size, :] = \
                        batch["image"][i].numpy()
                    cv2.imwrite("cancerous_red.jpg", batch["image"][i].numpy())

                    # if batch["label"][i] == 1:
                    #     cv2.imwrite("cancerous_red.jpg", masked)
                    #     print(cernainty)
                    #     print(pred[i])
                    #     print(batch["label"][i])
                    #     print(20*"---")
                    sys.stdout.flush()
                    sys.stdout.write("row:{r}, col:{c}\r".format(r=r, c=c))
                    # whole_slide_image[128 * r:128 * (r + 1), c * 128:(c + 1) * 128, :] = masked
                    whole_slide_image_normal[tile_size * r:tile_size * (r + 1), c * tile_size:(c + 1) * tile_size,
                    :] = \
                        batch["image"][i].numpy()

            # except:
            # print('err')
            # break
        save_dir = "./slides/tumor_{}.jpg".format(lamel)
        cv2.imwrite(save_dir, whole_slide_image)
        # cv2.imwrite("whole_slide_image_normal.jpg", whole_slide_image_normal)
        # df = df[df['std_img'] <= 10]
        # for row in df.iterrows():
        #     image_num = row["filename_img"].split("_")[-1].split(".")[0]
        #     i, j = np.unravel_index(image_num, shape=(row_count, col_count))
        #     whole_slide_image[i, j] = cv2.imread(os.path.join(data_path, row["filename"]))
    # except:
    #     print(lamel, "error")
    #     continue
