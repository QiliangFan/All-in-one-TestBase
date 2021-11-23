import SimpleITK as sitk
import os
from glob import glob
from scipy.ndimage import zoom, binary_fill_holes
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

# train_root = r"D:\dataset\promise2012\TrainData"
# output_root = r"D:\dataset\promise2012\0scaled_data"
# img_output = r"D:\dataset\promise2012\images"
# output_root = r"D:\dataset\promise2012\1corrected_data"
# img_output = r"D:\dataset\promise2012\corrected_images"
# seg_output = r"D:\dataset\promise2012\seg_images"

train_root = "/home/fanqiliang/data/TrainData"
# output_root = "/home/fanqiliang/data/processed_data/2normalized_data"
output_root = "/home/fanqiliang/data/processed_data/3denoised_data"

dst_size = np.asarray([128, 128, 64], dtype=np.float32)
dst_sp = np.asarray([1, 1, 1.5], dtype=np.float32)

def vnet_preprocess(img):
    space = np.asarray(img.GetSpacing())
    shape = np.asarray(img.GetSize())

    factor = space / dst_sp
    factorSize = np.asarray(shape * factor, dtype=float)
    newSize = np.max([factorSize, dst_size], axis=0).astype(dtype=np.int32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([dst_sp[0], dst_sp[1], dst_sp[2]])
    resampler.SetSize(newSize.tolist())
    resampler.SetInterpolator(sitk.sitkLinear)
    imgResampled = resampler.Execute(img)

    imgCentroid = np.asarray(newSize, dtype=float) / 2.0

    imgStartPx = (imgCentroid - dst_size / 2.0).astype(dtype='int')

    regionExtractor = sitk.RegionOfInterestImageFilter()
    size_2_set = dst_size.astype(dtype='int')
    regionExtractor.SetSize(size_2_set.tolist())
    regionExtractor.SetIndex(imgStartPx.tolist())

    imgResampledCropped = regionExtractor.Execute(imgResampled)

    return imgResampledCropped

def process(idx, file, seg):
    img = sitk.ReadImage(file)
    seg_img = sitk.ReadImage(seg)

    # denoise (不管用)
    # img = sitk.CurvatureFlow(image1=img,
    #             timeStep=0.125,
    #             numberOfIterations=5)

    img = sitk.Cast(img, sitk.sitkFloat32)
    seg_img = sitk.Cast(seg_img, sitk.sitkUInt8)

    # mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([200] * 4)
    img = corrector.Execute(img, seg_img)

    assert img.GetSize() == seg_img.GetSize()
    assert img.GetSpacing() == seg_img.GetSpacing()
    
    img = vnet_preprocess(img)
    seg_img = vnet_preprocess(seg_img)
    arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    #
    # resample
    #
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetReferenceImage(img)
    # resampler.SetOutputSpacing([seg_space[0] * seg_shape[0] / 128, seg_space[1] * seg_shape[1] / 128, seg_space[2] * seg_shape[2] / 65])
    # resampler.SetSize((128, 128, 65))
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    # seg_img = resampler.Execute(seg_img)
    # seg_arr = sitk.GetArrayFromImage(seg_img)[:-1]
    # img = resampler.Execute(img)
    # arr = sitk.GetArrayFromImage(img)[:-1]

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
    arr = (arr - mean) / std
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    sitk.WriteImage(sitk.GetImageFromArray(arr), os.path.join(output_root, f"{idx}.mhd"))
    sitk.WriteImage(sitk.GetImageFromArray(seg_arr), os.path.join(output_root, f"{idx}_seg.mhd"))

if __name__ == "__main__":
    seg_file = glob(os.path.join(train_root, "*_segmentation*.mhd"))
    file = [f.replace("_segmentation", "") for f in seg_file]
    arrs = np.concatenate([sitk.GetArrayFromImage(sitk.ReadImage(f)).flatten() for f in file], axis=0)
    
    mean = arrs[arrs > 0].mean()
    std = arrs[arrs > 0].std()

    with Pool(1) as pool:
        pool.starmap(process, [(idx, file[idx], seg_file[idx]) for idx in range(len(seg_file))])
