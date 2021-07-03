import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import cv2
import matplotlib
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import numpy as np
matplotlib.use('agg')

import paddlex as pdx
from paddlex.det import FasterRCNN, transforms
from glob import glob

test_img_root = "/home/maling/fanqiliang/data/coco_covid19/test_imgs"
train_dcm_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train"
img_level_csv = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"

train_transform = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(32)
])

test_transform = transforms.Compose([
    transforms.Normalize(),
    transforms.Resize(512),
    # transforms.ResizeByShort(short_size=800, max_size=1333),
    # transforms.Resize(),
])

train_dataset = pdx.datasets.CocoDetection(
    data_dir="/home/maling/fanqiliang/data/coco_covid19/imgs",
    ann_file="/home/maling/fanqiliang/data/coco_covid19/annotation.json",
    transforms = train_transform,
    shuffle=True,
    num_workers=8,
    buffer_size=8,
)
print("train_dataset...")

model = pdx.det.FasterRCNN(num_classes=len(train_dataset.labels)+1)
# model.train(
#     train_batch_size=1,
#     num_epochs=50,
#     train_dataset=train_dataset,
#     # eval_dataset=train_dataset,
#     learning_rate=0.0025,
#     lr_decay_epochs=[8, 11],
#     save_interval_epochs=1,
#     save_dir="output/faster_rcnn_r50_fpn",
#     use_vdl=True
# )

model: FasterRCNN = pdx.load_model("output/faster_rcnn_r50_fpn/epoch_50")


# ------------------------------- test ------------------------------------------------------------------- #
# images = glob(os.path.join(test_img_root, "*.jpeg"))
# for img in images:
#     result = model.predict(img, transforms=test_transform)
#     img = cv2.imread(img)
#     pdx.det.visualize(transforms.resize(img, 512), result, threshold=0.5, save_dir="./output/faster_rcnn_r50_fpn/imgs")
# -------------------------------------------------------------------------------------------------------- #

# ------------------------------- eval ------------------------------------------------------------------- #
from typing import Tuple, List
def parse_label(v: str) -> List:
    if "none" in v:
        return []
    else:
        b = v.split("opacity")
        b = [list(map(lambda x: float(x), i.strip().split(" ")))[1:]
             for i in b if len(i)]
        res = []
        for vec in b:
            res.append([vec[1], vec[0], vec[3], vec[2]])
        return res

dt = pd.read_csv(img_level_csv)
ids = dt["id"].apply(lambda v: v.replace("_image", "")).values
bboxes = dt["boxes"].apply(lambda v: v.replace(
    "'", "\"") if isinstance(v, str) else v).values
labels = dt["label"].values
study_instances = dt["StudyInstanceUID"].values
for _id, _label, _inst in zip(ids, labels, study_instances):
    bboxs = parse_label(_label)
    dcm = glob(os.path.join(train_dcm_root, "**", f"{_inst}", "**", f"{_id}.dcm"), recursive=True)[0]
    img: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(
            dcm, imageIO="GDCMImageIO")).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = np.asarray(Image.fromarray(img[0]).convert('RGB'))
    result = model.predict(img, transforms=test_transform)
    for y0, x0, y1, x1 in bboxs:
        cv2.rectangle(img, [int(x0), int(y0)], [int(x1), int(y1)], color=(255, 0, 0), thickness=10)
    pdx.det.visualize(transforms.resize(img, 512), result, threshold=0.5, save_dir="./output/faster_rcnn_r50_fpn/eval_imgs")
# -------------------------------------------------------------------------------------------------------- #
