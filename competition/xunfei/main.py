import os

from pytorch_lightning import callbacks
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import DataModule
from models.net import Net
import torch
from glob import glob

stage1_data_root = "/home/maling/fanqiliang/data/xunfei_image_recognition/stage_1"
labels = ["knife", "scissors", "sharpTools", "expandableBaton", "smallGlassBottle", "electricBaton",
          "plasticBeverageBottle", "plasticBottleWithaNozzle", "electronicEquipment", "battery", "seal", "umbrella"]


if __name__ == "__main__":
    labels_to_cls = {}
    for i, _label in enumerate(labels):
        labels_to_cls[_label] = i + 1   # background : 0

    datamodule = DataModule(labels_to_cls, os.path.join(
        stage1_data_root, "train"), os.path.join(stage1_data_root, "test"))

    ckpt_cb = ModelCheckpoint(dirpath="ckpt", save_weights_only=True)
    
    ckpt_files = glob(os.path.join("ckpt", "*.ckpt"))
    ckpt_files.sort()
    if len(ckpt_files) > 0:
        ckpt = ckpt_files[-1]
    else:
        ckpt = None

    trainer = Trainer(
        gpus=1 if torch.cuda.device_count() > 0 else 0, 
        max_epochs=20,
        callbacks=[ckpt_cb],
    )
    
    net = Net(num_classes=len(labels)+1, labels=labels, anchor_ratios=[0.0625, 0.125, 0.25, 0.3, 0.5, 1, 2, 3, 4, 8, 16], anchor_size=[1, 2, 3, 4, 8, 16, 32, 64, 128])
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt)["state_dict"])

    trainer.fit(net, datamodule=datamodule)
    trainer.test(net, datamodule=datamodule)