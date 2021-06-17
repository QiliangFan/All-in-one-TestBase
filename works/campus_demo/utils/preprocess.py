import os
from typing import Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import zoom
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import (binary_closing, binary_erosion,
                                convex_hull_image, disk)
from skimage.segmentation import clear_border
import pandas as pd
from visdom import Visdom
import time

"""
Global Var
"""
DST_SLICE_THICKNESS = 1


def normalize(ct: Union[np.ndarray, str]) -> np.ndarray:
    if isinstance(ct, str):
        ct: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(ct))
    ct = (ct - ct.min()) / (ct.max() - ct.min()) * 255
    return ct


def parenchyma_seg(ct: np.ndarray, vis: Visdom) -> np.ndarray:
    arr = ct.copy()

    # binarize
    threshold = -400
    arr = arr < threshold
    vis.image(arr.astype(np.float32), "1binarize")
    time.sleep(1)

    # clear border
    cleared: np.ndarray = clear_border(arr)

    vis.image(cleared.astype(np.float32), "2cleared")
    time.sleep(1)

    # divide into two areas (excluded with background)
    label_img = label(cleared)
    areas = [r.area for r in regionprops(label_img)]
    areas.sort()
    labels = []
    if len(areas) > 2:
        for region in regionprops(label_img):
            if region.area < areas[-2]:
                for x, y in region.coords:
                    label_img[x, y] = 0
    arr = label_img > 0
    vis.image(arr.astype(np.float32), "3two area")
    time.sleep(1)

    # fill holes
    arr = binary_erosion(arr, disk(2))
    arr = binary_closing(arr, disk(10))
    edges = roberts(arr)
    arr = ndimage.binary_fill_holes(edges)
    vis.image(arr.astype(np.float32), "4fill holes")
    time.sleep(1)

    res: np.ndarray = arr * ct
    vis.image((res.astype(np.float32) - res.min()) / (res.max() - res.min()) * 255, "seg result")
    time.sleep(1)

    return res


def gen_nodule(img: sitk.Image, nodule_csv: str, sid: str):
    import math
    nodule_csv: pd.DataFrame = pd.read_csv(nodule_csv)
    nodule_csv = nodule_csv[nodule_csv["seriesuid"] == sid]
    arr = sitk.GetArrayFromImage(img)
    nodule: np.ndarray = np.zeros_like(arr)
    for sid, x, y, z, d in nodule_csv.values:
        k, j, i = img.TransformPhysicalPointToIndex((x, y, z))
        space = img.GetSpacing()  
        space = space[::-1]  # (z, y, x)
        ri, rj, rk = math.ceil(d/space[0]/2), math.ceil(d/space[1]/2), math.ceil(d/space[2]/2)
        if 2 * ri * space[0] < 1: continue
        min_i, max_i = max(0, i-ri), min(i+ri, nodule.shape[0]-1)
        min_j, max_j = max(0, j-rj), min(j+rj, nodule.shape[1]-1)
        min_k, max_k = max(0, k-rk), min(k+rk, nodule.shape[2]-1)
        nodule[min_i:max_i, min_j:max_j, min_k:max_k] = 1
    return nodule