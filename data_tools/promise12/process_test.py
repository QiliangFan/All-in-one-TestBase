import SimpleITK as sitk
import os
from glob import glob
from scipy.ndimage import zoom, binary_fill_holes
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

test_root = "/home/fanqiliang/data/TestData"
output_root = "/home/fanqiliang/data/processed_data/processed_test"

if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)

def do(file: str):
    output_file = os.path.join(output_root, os.path.basename(file))

    img = sitk.ReadImage(file)
    img = sitk.Cast(img, sitk.sitkFloat32)
    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 8)
    img = corrector.Execute(img, mask_img)

    space = img.GetSpacing()
    shape = img.GetSize()
    
    arr = sitk.GetArrayFromImage(img)

    z_shape = arr.shape[0]
    for i in range(1, 10):
        if abs(z_shape - 2 ** i) < abs(z_shape - 2 ** (i+1)):
            z_size = 2 ** i
            break

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(
        [
            space[0] * shape[0] / 128, 
            space[1] * shape[1] / 128, 
            space[2] * shape[2] / (z_size+1)
        ]
    )
    resampler.SetSize((128, 128, (z_size+1)))

    img: sitk.Image = resampler.Execute(img)
    arr = sitk.GetArrayFromImage(img)[:-1]

    arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    output_img = sitk.GetImageFromArray(arr)
    output_img.SetDirection(img.GetDirection())
    output_img.SetSpacing(img.GetSpacing())
    sitk.WriteImage(output_img, output_file)
    print(file)
    return file

if __name__ == "__main__":
    test_file = glob(os.path.join(test_root, "*.mhd"))

    with Pool(processes=8) as pool:
        pool.map(do, test_file, chunksize=8)
        pool.close()
        pool.join()