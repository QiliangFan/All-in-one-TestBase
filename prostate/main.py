from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from data import DataModule
from net import Net
import torch


def main():
    data_module = DataModule(data_root, batch_size=4)

    net = Net()

    ckpt_model = ModelCheckpoint(dirpath="ckpt", save_weights_only=True, filename="net.ckpt")
    trainer = Trainer(gpus=1, max_epochs=1000, callbacks=[ckpt_model], log_every_n_steps=1, benchmark=True)

    # net.load_state_dict(torch.load("ckpt/epoch=125-step=1511.ckpt")["state_dict"])
    trainer.fit(net, datamodule=data_module)

    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = config["processed"]["train"]

    main()