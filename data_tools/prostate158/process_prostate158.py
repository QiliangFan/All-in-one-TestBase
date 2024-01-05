import os
import SimpleITK as sitk
from SimpleITK import Image
import numpy as np
from typing import Tuple
from glob import glob
from multiprocessing import Pool

origin_root = "/home/fanqiliang/data/prostate158_origin/train"
dst_root = "/home/fanqiliang/data/prostate158"
if not os.path.exists(dst_root):
    os.makedirs(dst_root)

SIZE = 128

def load_origin(file: str) -> Tuple[Image, Image]:
    origin_data_file = file
    origin_label_file = file.replace("t2.nii.gz", "t2_anatomy_reader1.nii.gz")

    label = sitk.ReadImage(origin_label_file)
    data = sitk.ReadImage(origin_data_file)
    label = sitk.Cast(label, sitk.sitkUInt8)
    data = sitk.Cast(data, sitk.sitkFloat32)
    label.SetOrigin(data.GetOrigin())
    label.SetSpacing(data.GetSpacing())
    label.SetDirection(data.GetDirection())
    return data, label

def process(data: Image, label: Image) -> Tuple[Image, Image]:
    # 1. 偏域修正
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 8)  # https://simpleitk.readthedocs.io/en/release/link_N4BiasFieldCorrection_docs.html
    data = corrector.Execute(data, label)

    # 重采样
    seg_space = label.GetSpacing()
    seg_shape = label.GetSize()
    z_shape = data.GetSize()[2]
    for i in range(1, 10):
        if abs(z_shape - 2 ** i) < abs(z_shape - 2 ** (i+1)):
            z_size = 2 ** i
            break
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(data)
    resampler.SetOutputSpacing([seg_space[0] * seg_shape[0] / SIZE, seg_space[1] * seg_shape[1] / SIZE, seg_space[2] * seg_shape[2] / (z_size+1)])
    resampler.SetSize((SIZE, SIZE, (z_size+1)))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_img = resampler.Execute(label)
    seg_arr = sitk.GetArrayFromImage(seg_img)[:-1]
    img = resampler.Execute(data)
    arr = sitk.GetArrayFromImage(img)[:-1]

    # 标准化
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    # arr = (arr - arr[arr > 0].mean()) / arr[arr > 0].std()
    seg_arr = np.where(seg_arr > 0.5, 1, 0).astype(int)
    return sitk.GetImageFromArray(arr), sitk.GetImageFromArray(seg_arr)

def work(idx, file):
    dst_label_file = os.path.join(dst_root, f"{idx}_seg.mhd")
    dst_data_file = os.path.join(dst_root, f"{idx}.mhd")
    data, label = load_origin(file)
    data, label = process(data, label)
    sitk.WriteImage(data, dst_data_file)
    sitk.WriteImage(label, dst_label_file)
    print(dst_data_file)

def main():
    origins = glob(os.path.join(origin_root, "*", "t2.nii.gz"))
    params = [(idx, file) for idx, file in enumerate(origins)]
    with Pool(processes=4) as pool:
        pool.starmap(work, params)
        

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()