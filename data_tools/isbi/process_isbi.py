import SimpleITK as sitk
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

isbi_origin = "/home/fanqiliang/data/ISBI13"
output_root = "/home/fanqiliang/datasets/prostate_wo_n4bc/isbi13"

label_map = {
    "CG": 2,
    "PZ": 1
}

if not os.path.exists(output_root):
    os.makedirs(output_root)

def do(idx: int, dcM_dir: str, seg_file: str, output_dir: str, cat: str):
    save_data_dir = os.path.join(output_dir, "data")
    save_label_dir = os.path.join(output_dir, "label")
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir, exist_ok=True)
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir, exist_ok=True)

    # read src image
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(dcM_dir)
    reader.SetFileNames(dcm_names)
    img = reader.Execute()

    seg_img = sitk.ReadImage(seg_file)
    seg_arr = sitk.GetArrayFromImage(seg_img)
    seg_arr[seg_arr != label_map[cat]] = 0
    seg_arr[seg_arr == label_map[cat]] = 1
    seg_img = sitk.GetImageFromArray(seg_arr)
    seg_img.SetSpacing(img.GetSpacing())
    seg_img.SetDirection(img.GetDirection())
    seg_img.SetOrigin(img.GetOrigin())

    if img.GetSize() != seg_img.GetSize():
        print(
            dcM_dir, 
            os.path.basename(seg_file),
            img.GetSize(),
            seg_img.GetSize()
        )

    img = sitk.Cast(img, sitk.sitkFloat32)
    # seg_img = sitk.Cast(seg_img, sitk.sitkUInt8)

    # corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # corrector.SetMaximumNumberOfIterations([50] * 8)
    # img = corrector.Execute(img, seg_img)

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
        os.path.join(save_data_dir, os.path.basename(f"{idx}.mhd")),
    )
    sitk.WriteImage(
        sitk.GetImageFromArray(seg_arr),
        os.path.join(save_label_dir, os.path.basename(f"{idx}.mhd"))
    )


if __name__ == "__main__":
    arr_files = glob(os.path.join(isbi_origin, "**", "*.dcm"), recursive=True)
    case_list = list(set([os.path.dirname(file) for file in arr_files]))

    seg_root = os.path.join(isbi_origin, "segment")

    idxs = []
    arr_dirs = []
    seg_files = []

    for i, dcm_dir in enumerate(case_list):
        series_id = os.path.basename(os.path.dirname(os.path.dirname(dcm_dir)))
        if series_id not in ["ProstateDx-01-0055"]:
            idxs.append(i)
            arr_dirs.append(dcm_dir)
            seg_files.append(os.path.join(seg_root, f"{series_id}.nrrd"))

    with Pool(16) as pool:
        for label_cat in ["CG", "PZ"]:
            output_dir = os.path.join(output_root, label_cat)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            pool.starmap(do, [(idx, arr_dir, seg_file, output_dir, label_cat) for idx, arr_dir, seg_file in zip(idxs, arr_dirs, seg_files)])
        pool.close()
        pool.join()
