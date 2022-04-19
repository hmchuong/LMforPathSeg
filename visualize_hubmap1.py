file_result = 'experiments/hubmap/contrast/val_dlv3_zeros.csv'
pred_dir = 'experiments/hubmap/contrast/outputs/unet_0413_uncon_128'
default_dir = 'experiments/hubmap/contrast/outputs/unet_0416_def'
images_dir = '/fs/class-projects/spring2022/cmsc828l/c828lg001/hubmap_patches/images'
label_dir = '/fs/class-projects/spring2022/cmsc828l/c828lg001/hubmap_patches/masks'

output_dir = "debug_unconnected_128_right"

import os
import cv2
import numpy as np

threshold = 0.1

os.makedirs(output_dir, exist_ok=True)

image_names = []
with open(file_result, "r") as f:
    for line in f:
        filename, iou = line.strip().split('\t')
        if float(iou) >= 0 and float(iou) < threshold:
            image_names.append(filename)

image_names = [
"c68fe75ea_005_018_image.jpg",
"b9a3865fc_011_002_image.jpg",
"e79de561c_002_000_image.jpg",
"e79de561c_002_009_image.jpg",
"e79de561c_001_003_image.jpg",
"b9a3865fc_011_012_image.jpg",
"e79de561c_002_010_image.jpg",
"c68fe75ea_011_020_image.jpg",
"c68fe75ea_012_001_image.jpg",
"b9a3865fc_003_003_image.jpg",
"cb2d976f4_009_018_image.jpg",
"e79de561c_003_000_image.jpg",
]

image_names = [
"c68fe75ea_010_017_image.jpg",
"cb2d976f4_008_004_image.jpg",
"c68fe75ea_010_018_image.jpg",
"e79de561c_004_011_image.jpg",
"c68fe75ea_011_016_image.jpg",
"b9a3865fc_011_018_image.jpg",
"b9a3865fc_003_015_image.jpg",
"c68fe75ea_011_004_image.jpg",
"e79de561c_006_004_image.jpg",
"e79de561c_001_004_image.jpg",
]

for image_name in image_names:
    pred = cv2.imread(os.path.join(pred_dir, image_name))
    defa = cv2.imread(os.path.join(default_dir, image_name))
    image = cv2.imread(os.path.join(images_dir, image_name))
    label = cv2.imread(os.path.join(label_dir, image_name.replace("_image", "_mask")))
    if len(pred.shape) == 2:
        pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        defa = np.repeat(defa[:, :, np.newaxis], 3, axis=2)
    if len(label.shape) == 2:
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)
    image = cv2.resize(image, (320, 320))
    label = cv2.resize(label, (320, 320))
    res = np.zeros((320, 320 * 4 + 60, 3),dtype=np.uint8)
    x = 0
    res[:, :, 0] = 255
    res[:, :x + 320] = image
    x += 340
    res[:, x: x + 320] = label
    x += 340
    res[:, x: x + 320] = defa
    x += 340
    res[:, x: x + 320] = pred
    cv2.imwrite(os.path.join(output_dir, image_name), res)
    