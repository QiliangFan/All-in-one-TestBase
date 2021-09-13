import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import List
import SimpleITK as sitk
from glob import glob
import os

class Data(Dataset):

    def __init__(self, arr_files: List, seg_files: List = None):
        super().__init__()
        assert seg_files is None or len(arr_files) == len(seg_files)

        self.arr_files = arr_files
        self.seg_files = seg_files

    def __getitem__(self, idx):
        arr_file = self.arr_files[idx]
        seg_file = self.seg_files[idx] if self.seg_files is not None else None

        arr = sitk.GetArrayFromImage(sitk.ReadImage(arr_file))
        arr = arr[None, :]

        if seg_file is not None:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
            seg = seg[None, :]
            arr, seg = torch.as_tensor(arr, dtype=torch.float32), torch.as_tensor(seg, dtype=torch.float32)
            return arr, seg
        else:
            arr = torch.as_tensor(arr, dtype=torch.float32)
            return arr

    def __len__(self):
        return len(self.arr_files)


class DataModule(LightningDataModule):

    def __init__(self, data_root: str, fold_num: int, batch_size = 2):
        super().__init__()

        self.batch_size = batch_size

        arr_files = glob(os.path.join(data_root, "*[0-9].mhd"))
        seg_files = [v.replace(".mhd", "_seg.mhd") for v in arr_files]
        assert len(arr_files) == len(seg_files)

        length = len(arr_files)
        test_len = length // 10
        train_len = length - test_len

        test_arrs = arr_files[(fold_num * test_len):((fold_num + 1) * test_len)]
        test_segs = seg_files[(fold_num * test_len):((fold_num + 1) * test_len)]

        train_arrs = list(filter(lambda v: v not in test_arrs, arr_files))
        train_segs = list(filter(lambda v: v not in test_segs, seg_files))
        
        self.train_data, self.test_data = Data(train_arrs, train_segs), Data(test_arrs, test_segs)


    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f"stage: {stage}")

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        return train_data

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True, num_workers=4)
        return test_data