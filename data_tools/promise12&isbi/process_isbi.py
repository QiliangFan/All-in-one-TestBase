import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

isbi_root = "/home/fanqiliang/data/isbi13"
output_root = "/home/fanqiliang/data/output_isbi13"

if not os.path.exists(output_root):
    os.makedirs(output_root)

def do(arr_file: str, seg_file: str, output_dir: str):
    data_dir = os.path.join(output_dir, "data")
    label_dir = os.path.join(output_dir, "label")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir, exist_ok=True)

    img = sitk.ReadImage(arr_file)
    seg_img = sitk.ReadImage(seg_file)
    seg_img.SetSpacing(img.GetSpacing())
    seg_img.SetDirection(img.GetDirection())
    seg_img.SetOrigin(img.GetOrigin())

    if img.GetSize() != seg_img.GetSize():
        print(
            os.path.basename(arr_file), 
            os.path.basename(seg_file),
            img.GetSize(),
            seg_img.GetSize()
            
        )
    # return

    img = sitk.Cast(img, sitk.sitkFloat32)
    seg_img = sitk.Cast(seg_img, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 8)
    img = corrector.Execute(img, seg_img)

    assert img.GetSize() == seg_img.GetSize()
    assert img.GetSpacing() == seg_img.GetSpacing()
    
    arr =  sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    arr_space = img.GetSpacing()
    arr_shape = img.GetSize()
    
    z_shape = arr_shape[2]
    for i in range(1, 10):
        if abs(z_shape - 2 ** i) < abs(z_shape - 2 ** (i+1)):
            z_size = 2 ** i
            break
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([arr_space[0] * arr_shape[0] / 128, arr_space[1] * arr_shape[1] / 128, arr_space[2] * arr_shape[2] / (z_size+1)])
    resampler.SetSize((128, 128, (z_size+1)))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    img = resampler.Execute(img)
    seg_img = resampler.Execute(seg_img)
    arr = sitk.GetArrayFromImage(img)[:-1]
    seg_arr = sitk.GetArrayFromImage(seg_img)[:-1]

    arr = (arr - arr.min()) / (arr.max() - arr.min())
    seg_arr = np.where(seg_arr > 0.5, 1, 0).astype(int)

    sitk.WriteImage(
        sitk.GetImageFromArray(arr),
        os.path.join(data_dir, os.path.basename(arr_file)),
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(seg_arr),
        os.path.join(label_dir, os.path.basename(seg_file))
    )


if __name__ == "__main__":
    arr_files = glob(os.path.join(isbi_root, "data", "*.mhd"))

    for category in ["PZ", "CG"]:
        label_root = os.path.join(isbi_root, f"{category}_label")
        seg_files = [f.replace(os.path.join(isbi_root, "data"), label_root) for f in arr_files]
        output_dir = os.path.join(output_root, category)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with Pool(32) as pool:
            pool.starmap(do, [(l, s, output_dir) for l, s in zip(arr_files, seg_files)])
            pool.close()
            pool.join()

