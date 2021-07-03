from typing import List, Tuple
import cv2
from PIL import Image
import SimpleITK as sitk
import os
from glob import glob
import pandas as pd
import numpy as np
import torch
from visdom import Visdom
from utils.preprocess import normalize, parenchyma_seg
import time
import math


def get_ct_with_nodule(nodule_csv: str):
    dt = pd.read_csv(nodule_csv)
    sids = dt["seriesuid"].values
    coordX = dt["coordX"].values
    coordY = dt["coordY"]
    coordZ = dt["coordZ"]
    diameter_mm = dt["diameter_mm"]

    for sid, x, y, z, d in zip(sids, coordX, coordY, coordZ, diameter_mm):
        mhd_file = glob(os.path.join(
            ct_root, "**", f"{sid}.mhd"), recursive=True)[0]
        img = sitk.ReadImage(mhd_file)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        raw = normalize(arr)

        (_x, _y, _z) = img.TransformPhysicalPointToIndex((x, y, z))
        if vis is not None:
            vis.image(raw[_z], win="0raw")
            time.sleep(1)  # wait for show

        show_res = raw[_z]
        show_res = Image.fromarray(show_res).convert(mode="RGB")
        show_res = np.asarray(show_res).astype(np.float32)
        cv2.rectangle(show_res, (math.ceil(_x - d//2 - 5), math.ceil(_y - d//2 -5)),
                      (math.ceil(_x + d//2 + 5), math.ceil(_y + d//2 + 5)), (0, 255, 0), thickness=2)
        print(show_res.shape, show_res.dtype)

        if vis is not None:
            vis.image(show_res.transpose(2, 0, 1), "5final result")
            time.sleep(5)

        parenchyma_seg(arr[_z], vis)


def main():
    get_ct_with_nodule(nodule_csv)


if __name__ == "__main__":
    nodule_csv = "/home/maling/fanqiliang/lung16/CSVFILES/annotations.csv"
    ct_root = "/home/maling/fanqiliang/lung16/LUNG16"

    # vis = Visdom()
    vis = None
    main()
