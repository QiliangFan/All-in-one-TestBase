from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from data import DataModule
from net import Net


def main():
    data_module = DataModule(data_root)

    net = Net()

    ckpt_model = ModelCheckpoint(dirpath="ckpt", save_weights_only=True)
    trainer = Trainer(max_epochs=5, callbacks=[ckpt_model], benchmark=True)

    trainer.fit(net, datamodule=data_module)

    trainer.test(net, datamodule=data_module)


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = config["processed"]["train"]

    main()