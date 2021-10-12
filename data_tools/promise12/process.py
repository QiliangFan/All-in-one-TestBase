import SimpleITK as sitk
import os
from glob import glob
from scipy.ndimage import zoom, binary_fill_holes
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

train_root = "/home/lisiyi/data/Prostate/TrainData"
output_root = "/home/lisiyi/processed_data0scaled_data"
# output_root = r"/home/lisiyi/processed_data/1corrected_data"


# img_output = r"D:\dataset\promise2012\corrected_images"
# seg_output = r"D:\dataset\promise2012\seg_images"
# img_output = r"D:\dataset\promise2012\images"



def process(idx, file, seg):
    img = sitk.ReadImage(file)
    img = sitk.Cast(img, sitk.sitkFloat32)

    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 4)
    img = corrector.Execute(img, mask_img)

    seg_img = sitk.ReadImage(seg)
    assert img.GetSize() == seg_img.GetSize()
    assert img.GetSpacing() == seg_img.GetSpacing()

    space = img.GetSpacing()
    shape = img.GetSize()

    seg_space = seg_img.GetSpacing()
    seg_shape = seg_img.GetSize()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([seg_space[0] * seg_shape[0] / 128, seg_space[1] * seg_shape[1] / 128, seg_space[2] * seg_shape[2] / 65])
    resampler.SetSize((128, 128, 65))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_img = resampler.Execute(seg_img)
    seg_arr = sitk.GetArrayFromImage(seg_img)[:-1]
    img = resampler.Execute(img)
    arr = sitk.GetArrayFromImage(img)[:-1]

    # arr = zoom(arr, (64 / arr.shape[0], 128 / arr.shape[1], 128 / arr.shape[2])).astype(np.float32)
    # seg_arr = zoom(seg_arr, (64 / seg_arr.shape[0], 128 / seg_arr.shape[1], 128 / seg_arr.shape[2]), mode='nearest')
    # seg_arr = np.where(seg_arr > 0.5, 1, 0).astype(int)

    # for i, slice in enumerate(arr):
    #     fig = plt.figure()
    #     plt.axis("off")
    #     plt.imshow(slice)
    #     # plt.savefig(os.path.join(img_output, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.png"), bbox_inches="tight")
    #     plt.savefig(os.path.join(img_output, f"{idx}_{i}.png"), bbox_inches="tight")
    #     plt.close(fig)

    # arr = (arr - arr[arr > 0].mean()) / arr[arr > 0].std()

    sitk.WriteImage(sitk.GetImageFromArray(arr), os.path.join(output_root, f"{idx}.mhd"))
    sitk.WriteImage(sitk.GetImageFromArray(seg_arr), os.path.join(output_root, f"{idx}_seg.mhd"))

if __name__ == "__main__":
    seg_file = glob(os.path.join(train_root, "*_segmentation*.mhd"))
    file = [f.replace("_segmentation", "") for f in seg_file]

    with Pool(8) as pool:
        pool.starmap(process, [(idx, file[idx], seg_file[idx]) for idx in range(len(seg_file))])
