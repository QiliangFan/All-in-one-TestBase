from pytorch_lightning import Trainer
from fast_rcnn_vgg import FasterRCNNVGG
from net import Net
from dataset import ImageLevelData

train_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train"
image_level = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"

trainer = Trainer(gpus=1, max_epochs=1, fast_dev_run=False)
model = FasterRCNNVGG(n_fg_class=1)
net = Net(model)
data = ImageLevelData(train_root, image_level)
trainer.fit(net, datamodule=data)