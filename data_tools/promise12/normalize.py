import SimpleITK as sitk
import os
from glob import glob
from scipy.ndimage import zoom, binary_fill_holes
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from nyul import nyul_train_standard_scale, nyul_apply_standard_scale

train_root = r"D:\dataset\promise2012\1corrected_data"
output_root = r"D:\dataset\promise2012\2normalized_data"
img_output = r"D:\dataset\promise2012\normalized_images"


def normalize():
    seg_mr = glob(os.path.join(train_root, "*seg*.mhd"))
    train_mr = [f.replace("_seg", "") for f in seg_mr]

    standard_scale, perc = nyul_train_standard_scale(train_mr, seg_mr)
    np.save("standard_hist.npy", [standard_scale, perc])

    for idx, (mr, seg) in enumerate(zip(train_mr, seg_mr)):
        image = sitk.GetArrayFromImage(sitk.ReadImage(mr))
        image_norm = nyul_apply_standard_scale(image, "standard_hist.npy")

        for i, slice in enumerate(image_norm):
            plt.figure()
            plt.axis("off")
            plt.imshow(slice, cmap="bone")
            plt.savefig(f"{img_output}/{idx}_{i}.png", bbox_inches="tight")
            plt.close()

        sitk.WriteImage(sitk.GetImageFromArray(image_norm), os.path.join(output_root, f"{idx}.mhd"))
        sitk.WriteImage(sitk.ReadImage(seg), os.path.join(output_root, f"{idx}_seg.mhd"))



if __name__ == "__main__":

    normalize()


