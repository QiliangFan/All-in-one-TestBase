import cv2
import matplotlib
from paddlex.cv.transforms.det_transforms import Resize
matplotlib.use('agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import paddlex as pdx
from paddlex.det import FasterRCNN, transforms

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
#     num_epochs=2,
#     train_dataset=train_dataset,
#     # eval_dataset=train_dataset,
#     learning_rate=0.0025,
#     lr_decay_epochs=[8, 11],
#     save_interval_epochs=1,
#     save_dir="output/faster_rcnn_r50_fpn",
#     use_vdl=True
# )

model: FasterRCNN = pdx.load_model("output/faster_rcnn_r50_fpn/epoch_2")
# model = pdx.load_model("output/faster_rcnn_r50_fpn/best_model")

image_name = "/home/maling/fanqiliang/data/coco_covid19/imgs/823201a39872-6564f55c1619.jpeg"
image_name = "/home/maling/fanqiliang/data/coco_covid19/imgs/81a0dd29620e-db4de26ecd1a.jpeg"
result = model.predict(image_name, transforms=test_transform)
img = cv2.imread(image_name)
# pdx.det.visualize(image_name, result, threshold=0.5, save_dir="./output/faster_rcnn_r50_fpn")
pdx.det.visualize(transforms.resize(img, 512), result, threshold=0.5, save_dir="./output/faster_rcnn_r50_fpn")
