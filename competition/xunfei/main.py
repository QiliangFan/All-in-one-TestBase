import os
from pytorch_lightning import Trainer
from dataset import DataModule
from models.net import Net

stage1_data_root = "/home/maling/fanqiliang/data/xunfei_image_recognition/stage_1"
labels = ["knife", "scissors", "sharpTools", "expandableBaton", "smallGlassBottle", "electricBaton",
          "plasticBeverageBottle", "plasticBottleWithaNozzle", "electronicEquipment", "battery", "seal", "umbrella"]


if __name__ == "__main__":
    labels_to_cls = {}
    for i, _label in enumerate(labels):
        labels_to_cls[_label] = i + 1   # background : 0

    datamodule = DataModule(labels_to_cls, os.path.join(
        stage1_data_root, "train"), os.path.join(stage1_data_root, "test"))

    trainer = Trainer(
        gpus=1, 
        max_epochs=5
    )

    net = Net(num_classes=len(labels)+1, labels=labels)

    trainer.fit(net, datamodule=datamodule)