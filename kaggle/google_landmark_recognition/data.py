from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import cv2
from typing import List

class LandMark(Dataset):

    def __init__(self, files: List, labels: List = None):
        super().__init__()

        self.files = files
        self.labels = labels

    def __getitem__(self, item):
        arr = cv2.imread(self.files[item])
        arr: np.ndarray = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr = arr.transpose()
        arr = torch.from_numpy(arr)
        if self.labels is not None:
            label = self.labels[item]
            label = torch.as_tensor(label)
            return arr, label
        else:
            return arr

    def __len__(self):
        return len(self.files)

class LandMarkDataModule(LightningDataModule):

    def __init__(self, train_files: List, test_files: List, labels: List):
        super().__init__()

        self.train_data = LandMark(train_files, labels)
        self.test_data = LandMark(test_files)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f"stage: {stage}")

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, num_workers=4, pin_memory=True, batch_size=32, shuffle=True)
        return train_data

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, num_workers=4, pin_memory=True, batch_size=32, shuffle=True)
        return test_data