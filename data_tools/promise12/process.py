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
# output_root = "/home/fanqiliang/data/processed_data/v2" # 偏域修正 + 归一化
# output_root = "/home/fanqiliang/data/processed_data/v3" # 单张图片标准化
# output_root = "/home/fanqiliang/data/processed_data/v4" # 偏域修正 + 单张图片标准化 + 归一化
# output_root = "/home/fanqiliang/data/processed_data/hist"  # 直方图均衡
output_root = "/home/fanqiliang/data/processed_data/hist_32"  # 直方图均衡


SIZE = 128
Z_SIZE = 64

# output_root = "/home/fanqiliang/data/processed_data/2normalized_data"
# output_root = "/home/fanqiliang/data/processed_data/1self_std"

if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)

def process(idx, file, seg):
    img = sitk.ReadImage(file)
    seg_img = sitk.ReadImage(seg)

    # denoise (不管用)
    # img = sitk.CurvatureFlow(image1=img,
    #             timeStep=0.125,
    #             numberOfIterations=5)

    img = sitk.Cast(img, sitk.sitkFloat32)
    seg_img = sitk.Cast(seg_img, sitk.sitkUInt8)

    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 8)  # https://simpleitk.readthedocs.io/en/release/link_N4BiasFieldCorrection_docs.html
    img = corrector.Execute(img, seg_img)

    # 直方图均衡
    # filter = sitk.AdaptiveHistogramEqualizationImageFilter()
    # img = filter.Execute(img)

    assert img.GetSize() == seg_img.GetSize()
    assert img.GetSpacing() == seg_img.GetSpacing()
    
    arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg_img)
    # assert arr.shape[2] == dst_size[0] and arr.shape[1] == dst_size[1] and arr.shape[0] == dst_size[2]
    seg_space = seg_img.GetSpacing()
    seg_shape = seg_img.GetSize()

    z_shape = arr.shape[0]
    for i in range(1, 10):
        if abs(z_shape - 2 ** i) < abs(z_shape - 2 ** (i+1)):
            z_size = 2 ** i
            break
    # z_size = 32
    #
    # resample
    #
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([seg_space[0] * seg_shape[0] / SIZE, seg_space[1] * seg_shape[1] / SIZE, seg_space[2] * seg_shape[2] / (z_size+1)])
    resampler.SetSize((SIZE, SIZE, (z_size+1)))
    
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_img = resampler.Execute(seg_img)
    seg_arr = sitk.GetArrayFromImage(seg_img)[:-1]
    img = resampler.Execute(img)
    arr = sitk.GetArrayFromImage(img)[:-1]


    # arr = zoom(arr, (64 / arr.shape[0], SIZE / arr.shape[1], SIZE / arr.shape[2])).astype(np.float32)
    # seg_arr = zoom(seg_arr, (64 / seg_arr.shape[0], SIZE / seg_arr.shape[1], SIZE / seg_arr.shape[2]), mode='nearest')

    # for i, slice in enumerate(arr):
    #     fig = plt.figure()
    #     plt.axis("off")
    #     plt.imshow(slice)
    #     # plt.savefig(os.path.join(img_output, f"{os.path.splitext(os.path.basename(file))[0]}_{i}.png"), bbox_inches="tight")
    #     plt.savefig(os.path.join(img_output, f"{idx}_{i}.png"), bbox_inches="tight")
    #     plt.close(fig)


    mean = arr.mean()
    std = arr.std()

    # arr = (arr - mean) / std
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    # arr = (arr - arr[arr > 0].mean()) / arr[arr > 0].std()
    seg_arr = np.where(seg_arr > 0.5, 1, 0).astype(int)

    sitk.WriteImage(sitk.GetImageFromArray(arr), os.path.join(output_root, f"{idx}.mhd"))
    sitk.WriteImage(sitk.GetImageFromArray(seg_arr), os.path.join(output_root, f"{idx}_seg.mhd"))
    # return arr.flatten()

if __name__ == "__main__":
    seg_file = glob(os.path.join(train_root, "*_segmentation*.mhd"))
    file = [f.replace("_segmentation", "") for f in seg_file]
    arrs = np.concatenate([sitk.GetArrayFromImage(sitk.ReadImage(f)).flatten() for f in file], axis=0)
    
    # mean = arrs[arrs > 0].mean()
    # std = arrs[arrs > 0].std()
    # mean = 281.41382
    # std = 340.0665

    with Pool(1) as pool:
        pool.starmap(process, [(idx, file[idx], seg_file[idx]) for idx in range(len(seg_file))])
        pool.close()
        pool.join()
        # result = pool.starmap(process, [(idx, file[idx], seg_file[idx]) for idx in range(len(seg_file))])
    # r = np.concatenate(result)
    # print(r.mean(), r.std())