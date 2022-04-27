
pred_dir = 'experiments/camelyon/outputs/nocontrast'
default_dir = 'experiments/camelyon/outputs/contrast'
images_dir = '/fs/class-projects/spring2022/cmsc828l/c828lg001/camelyon16/lamels'
label_dir = '/fs/class-projects/spring2022/cmsc828l/c828lg001/camelyon16/lamels'

output_dir = "camelyon_viz"

import os
import cv2
import numpy as np
import pandas as pd

threshold = 0.1

os.makedirs(output_dir, exist_ok=True)

image_names = [
"tumor_014_img_06853.jpg",
"tumor_014_img_06075.jpg",
"tumor_014_img_07240.jpg",
"tumor_014_img_06958.jpg",
"tumor_014_img_06671.jpg",
"tumor_014_img_06574.jpg",
"tumor_014_img_06377.jpg",
"tumor_014_img_06170.jpg",
"tumor_014_img_06170.jpg",
"tumor_014_img_07045.jpg",
"tumor_014_img_06662.jpg",
"tumor_014_img_06853.jpg",
"tumor_014_img_06364.jpg",
"tumor_014_img_06364.jpg",
"tumor_014_img_06077.jpg",
"tumor_014_img_06075.jpg",
"tumor_014_img_06075.jpg",
"tumor_014_img_06075.jpg",
"tumor_014_img_06567.jpg",
"tumor_014_img_06853.jpg",
"tumor_014_img_06853.jpg",
"tumor_014_img_06364.jpg",
"tumor_014_img_06077.jpg",
"tumor_014_img_06077.jpg",
"tumor_014_img_07052.jpg",
"tumor_014_img_06075.jpg",
"tumor_014_img_07240.jpg",
"tumor_014_img_07240.jpg",
"tumor_014_img_07147.jpg",
"tumor_014_img_07052.jpg",
"tumor_014_img_06076.jpg",
"tumor_014_img_06853.jpg",
"tumor_014_img_06456.jpg",
"tumor_014_img_06077.jpg",
"tumor_014_img_06861.jpg"
]

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

for image_name in image_names:
    print(image_name)
    pred = cv2.imread(os.path.join(pred_dir, image_name))
    defa = cv2.imread(os.path.join(default_dir, image_name))
    image = cv2.imread(os.path.join(images_dir, image_name))
    rle = pd.read_pickle(os.path.join(label_dir, image_name.replace(".jpg", "").replace("_img_", "_rle_")))
    label = rle2mask(rle, (1024, 1024))
    label = label * 255
    if len(pred.shape) == 2:
        pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
        defa = np.repeat(defa[:, :, np.newaxis], 3, axis=2)
    if len(label.shape) == 2:
        label = np.repeat(label[:, :, np.newaxis], 3, axis=2)
    image = cv2.resize(image, (320, 320))
    label = cv2.resize(label, (320, 320))
    defa = cv2.resize(defa, (320, 320))
    pred = cv2.resize(pred, (320, 320))
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
    