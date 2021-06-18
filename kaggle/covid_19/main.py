from pytorch_lightning import Trainer
from fast_rcnn_vgg import FasterRCNNVGG
from net import Net
from dataset import ImageLevelData
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
import os


train_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train"
image_level = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"
base_path = os.path.dirname(os.path.abspath(__file__))

ckpts = glob(os.path.join(base_path, "ckpt", "*.ckpt"))
if len(ckpts) > 0:
    ckpts.sort()
    ckpt_model = ckpts[-1]
else:
    ckpt_model = None
ckpt = ModelCheckpoint(dirpath="ckpt")
trainer = Trainer(gpus=1, max_epochs=10, fast_dev_run=False, callbacks=[ckpt], resume_from_checkpoint=ckpt_model)
model = FasterRCNNVGG(n_fg_class=1, ratios=[0.25, 1, 4], anchor_scales=[4, 8, 16])
net = Net(model)
data = ImageLevelData(train_root, image_level)
trainer.fit(net, datamodule=data)
trainer.test(net, datamodule=data)