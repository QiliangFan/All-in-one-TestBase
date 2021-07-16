import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from utils import load_train_data
from typing import Optional, Sequence
import cv2
import numpy as np
from glob import glob
import os


class Data(Dataset):

    def __init__(self, label2cls: dict, imgs: Sequence[str], bboxs: Optional[Sequence[dict]] = None):
        super().__init__()

        self.label2cls = label2cls
        self.imgs = imgs
        self.bboxs = bboxs

    def __getitem__(self, idx):
        img_path: str = self.imgs[idx]
        bboxs = []   # [y_min, x_min, height, width, cls]
        if self.bboxs is not None:
            bbox = self.bboxs[idx]
            for label, _bbox in bbox.items():
                cls = self.label2cls[label]
                for y_min, x_min, height, width in _bbox:
                    bboxs.append([y_min, x_min, height, width, cls])
        arr: np.ndarray = cv2.imread(img_path)
        arr = arr.transpose([2, 0, 1]) # (C, H, W)
        arr: torch.Tensor = torch.as_tensor(arr, dtype=torch.float32).flip(dims=(0,))  # (RGB)
        bboxs = torch.as_tensor(bboxs)
        return arr, bboxs

    def __len__(self):
        return len(self.imgs)


class DataModule(LightningDataModule):

    def __init__(self, label2cls: dict, train_root: str, test_root: str):
        super().__init__()

        imgs, bounding_box = load_train_data(train_root)
        self.train_data = Data(label2cls, imgs, bounding_box)

        imgs = glob(os.path.join(test_root, "*.jpg"))
        self.test_data = Data(label2cls, imgs)
        

    def setup(self, stage: str):
        print(f"Stage: {stage}")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        train_data = DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)
        return train_data

    def test_dataloader(self) -> DataLoader:
        test_data = DataLoader(self.test_data, batch_size=1, num_workers=8, pin_memory=True, prefetch_factor=8)
        return test_data