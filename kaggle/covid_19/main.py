import os
from glob import glob
import random
import torch
random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

import autogluon.core as ag
import autogluon
from autogluon.core import scheduler as sc
from autogluon.core.scheduler.reporter import LocalStatusReporter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ImageLevelData
from net import Net

data_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection"
image_level = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"
base_path = os.path.dirname(os.path.abspath(__file__))

@ag.args(
    mid_channel = ag.space.Categorical(128, 256, 512, 1024),
    lr = ag.space.Real(1e-6, 0.1, log=True),
    weight_decay = ag.space.Real(1e-12, 1e-3, log=True),
    epochs=10,
)
def search(args, reporter: LocalStatusReporter):
    train(args.mid_channel, args.lr, args.weight_decay, args.epochs, reporter)

def train(mid_channel = 512, lr = 1e-3, weight_decay = 1e-9, epochs = 50, reporter: LocalStatusReporter = None):
    ckpts = glob(os.path.join(base_path, "ckpt", "*.ckpt"))
    if len(ckpts) > 0:
        ckpts.sort()
        ckpt_model = ckpts[-1]
    else:
        ckpt_model = None
    ckpt = ModelCheckpoint(dirpath="ckpt")
    trainer = Trainer(
        gpus=1, 
        max_epochs=epochs, 
        fast_dev_run=False, 
        callbacks=[ckpt], 
        resume_from_checkpoint=ckpt_model,
        gradient_clip_val=1e6,
        deterministic=True
    )
    net = Net(mid_channel, lr, weight_decay, reporter)
    data = ImageLevelData(data_root, image_level)

    trainer.fit(net, datamodule=data)
    # trainer.test(net, datamodule=data)

if __name__ == "__main__":
    train(epochs=500, lr=0.0005, weight_decay=1e-9, mid_channel=128)
    # with autogluon.utils.warning_filter():
    #     scheduler = sc.FIFOScheduler(search,
    #         num_trials = 50,
    #         resource={"num_gpus": 2}, 
    #         time_attr="epoch",
    #         reward_attr="accuracy")
    #     scheduler.run()
    #     scheduler.join_jobs()

    # print(f"Best config: {scheduler.get_best_config()}")
    # print(f"Best reward: {scheduler.get_best_reward()}")
