from matplotlib.pyplot import axis
import torch
import numpy as np
import SimpleITK as sitk
import os
from glob import glob
from multiprocessing import Pool

train_root = "/home/fanqiliang/data/TrainData"
promise_aug_root = "/home/fanqiliang/data/promise_aug"
if not os.path.exists(promise_aug_root):
    os.makedirs(promise_aug_root, exist_ok=True)


def process(idx, file, seg):
    img = sitk.ReadImage(file)
    seg_img = sitk.ReadImage(seg)

    arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    flip_op = [
        0,
        1,
        2
    ]
    for op in flip_op:
        arr = np.flip(arr, axis=op)
        seg_arr = np.flip(seg_arr, axis=op)
    img = sitk.GetImageFromArray(arr)
    seg_img = sitk.GetImageFromArray(seg_arr)

    sitk.WriteImage(img, os.path.join(promise_aug_root, f"{idx}.mhd"))
    sitk.WriteImage(seg_img, os.path.join(promise_aug_root, f"{idx}_seg.mhd"))



if __name__ == "__main__":
    seg_file = glob(os.path.join(train_root, "*_segmentation*.mhd"))    
    file = [f.replace("_segmentation", "") for f in seg_file]
    arrs = np.concatenate([sitk.GetArrayFromImage(sitk.ReadImage(f)).flatten() for f in file], axis=0)
    
    # mean = arrs[arrs > 0].mean()
    # std = arrs[arrs > 0].std()
    # mean = 281.41382
    # std = 340.0665

    with Pool(16) as pool:
        pool.starmap(process, [(idx, file[idx], seg_file[idx]) for idx in range(len(seg_file))])