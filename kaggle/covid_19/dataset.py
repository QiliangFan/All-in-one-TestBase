from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import List, Tuple
from glob import glob
import os
import SimpleITK as sitk
import torch
import numpy as np
from scipy.ndimage import zoom
sitk.ProcessObject_GlobalWarningDisplayOff()

def parse_label(v: str) -> Tuple[List, List]:
    if "none" in v:
        return [0], []
    else:
        b = v.split("opacity")
        b = [list(map(lambda x: float(x), i.strip().split(" ")))[1:]
             for i in b if len(i)]
        res = []
        for vec in b:
            res.append([vec[1], vec[0], vec[3], vec[2]])
        return [1], res


class ImageData(Dataset):

    def __init__(self, data_root: str, csv: str = None):
        super().__init__()
        self.data_root = data_root
        self.csv = csv

        if self.csv is not None:
            dt = pd.read_csv(self.csv)
            dt = dt[~dt["label"].str.contains("none")]
            self.ids = dt["id"].apply(lambda v: v.replace("_image", "")).values
            self.bboxes = dt["boxes"].apply(lambda v: v.replace(
                "'", "\"") if isinstance(v, str) else v).values
            self.labels = dt["label"].values
            self.study_instances = dt["StudyInstanceUID"].values

        else:
            self.files = glob(os.path.join(
                self.data_root, "**", "*.dcm"), recursive=True)

    def __getitem__(self, idx: int):
        scale = 4
        if self.csv is not None:
            _id = self.ids[idx]
            _bbox = self.bboxes[idx]
            _label = self.labels[idx]
            _study_instance = self.study_instances[idx]

            label, bboxs = parse_label(_label)
            img_path = glob(os.path.join(
                self.data_root, "**", f"{_study_instance}", "**", f"{_id}.dcm"), recursive=True)[0]
            img: np.ndarray = sitk.GetArrayFromImage(
                sitk.ReadImage(img_path, imageIO="GDCMImageIO")).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = zoom(img, zoom=(1, 1/scale, 1/scale))

            return torch.as_tensor(img), torch.as_tensor(bboxs), torch.as_tensor(label), scale, _id, _study_instance
        else:
            img_path = self.files[idx]
            _id = os.path.splitext(os.path.basename(img_path))[0]
            _study_instance = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            img: np.ndarray = sitk.GetArrayFromImage(
                sitk.ReadImage(img_path)).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = zoom(img, zoom=(1, 1/scale, 1/scale))
            
            return torch.as_tensor(img), scale, _id, _study_instance

    def __len__(self):
        if self.csv is not None:
            return len(self.ids)
        else:
            return len(self.files)


class ImageLevelData(LightningDataModule):

    def __init__(self, data_root: str, img_label_csv: str):
        super().__init__()

        train_root = os.path.join(data_root, "train")
        test_root = os.path.join(data_root, "test")
        self.train_data = ImageData(train_root, img_label_csv)
        self.test_data = ImageData(test_root)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        """[summary]

        Args:
            stage (str): trainer 调用的函数名
        """
        print(f"Stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=8, pin_memory=True, prefetch_factor=8)
