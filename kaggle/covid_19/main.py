import os
from glob import glob

import autogluon.core as ag
import autogluon
from autogluon.core import scheduler as sc
from autogluon.core.scheduler.reporter import LocalStatusReporter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import ImageLevelData
from net import Net

train_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train"
image_level = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"
base_path = os.path.dirname(os.path.abspath(__file__))

@ag.args(
    mid_channel = ag.space.Categorical(128, 256, 512, 1024),
    lr = ag.space.Real(1e-6, 0.1, log=True),
    weight_decay = ag.space.Real(1e-12, 1e-4, log=True),
    epochs=10,
)
def search(args, reporter: LocalStatusReporter):
    train(args.mid_channel, args.lr, args.weight_decay, args.epochs, reporter)

def train(mid_channel = 512, lr = 1e-4, weight_decay = 1e-8, epochs = 1, reporter: LocalStatusReporter = None):
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
        # callbacks=[ckpt], 
        # resume_from_checkpoint=ckpt_model
    )
    net = Net(mid_channel, lr, weight_decay, reporter)
    data = ImageLevelData(train_root, image_level)
    trainer.fit(net, datamodule=data)
    # trainer.test(net, datamodule=data)

if __name__ == "__main__":

    with autogluon.utils.warning_filter():
        scheduler = sc.FIFOScheduler(search,
            num_trials = 50,
            resource={"num_gpus": 2}, 
            time_attr="epoch",
            reward_attr="accuracy")
        scheduler.run()
        scheduler.join_jobs()

    print(f"Best config: {scheduler.get_best_config()}")
    print(f"Best reward: {scheduler.get_best_reward()}")
