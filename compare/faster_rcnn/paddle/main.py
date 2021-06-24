import matplotlib
matplotlib.use('agg')
import os
os.environ["CUDA_VISIBLE_DEVIECES"] = '0'
import paddlex as pdx

class Dataset():



num_classes = 2
model = pdx.det.FasterRCNN(num_classes=2)
model.train(
    num_epochs=12,
    train_dataset=
)