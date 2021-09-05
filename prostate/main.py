from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from data import DataModule
from net import Net
import torch
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    data_module = DataModule(data_root, batch_size=4)

    net = Net()
    net.apply(weights_init)

    ckpt_model = ModelCheckpoint(dirpath="ckpt", save_weights_only=True, filename="net", monitor="dice", mode="max")
    trainer = Trainer(gpus=1, max_epochs=5000, callbacks=[ckpt_model], log_every_n_steps=1, benchmark=True)


    net.load_state_dict(torch.load("ckpt/net.ckpt")["state_dict"])
    trainer.fit(net, datamodule=data_module)

    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = config["processed"]["train"]

    main()