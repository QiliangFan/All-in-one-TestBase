import os

from pytorch_lightning import callbacks
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import DataModule
from models.net import Net
import torch

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
    # ckpt = None

    trainer = Trainer(
        gpus=1 if torch.cuda.device_count() > 0 else 0, 
        max_epochs=20,
        callbacks=[ckpt_cb],
        # resume_from_checkpoint=ckpt
    )
    
    net = Net(num_classes=len(labels)+1, labels=labels, anchor_ratios=[0.0625, 0.125, 0.25, 0.3, 0.5, 1, 2, 3, 4, 8, 16], anchor_size=[1, 2, 3, 4, 8, 16, 32, 64, 128])
    # net = Net(num_classes=len(labels)+1, labels=labels, anchor_ratios=[0.25, 0.5, 1, 2, 4], anchor_size=[0.5, 2, 4, 8, 16])

    trainer.fit(net, datamodule=datamodule)
    trainer.test(net, datamodule=datamodule)