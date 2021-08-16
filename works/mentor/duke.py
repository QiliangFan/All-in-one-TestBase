data_root = "/home/maling/fanqiliang/data/Duke-Breast-Cancer-MRI"
save_root = "/home/maling/fanqiliang/output_data/medical_data/mr"

import os
import shutil

annotation_csv = os.path.join(data_root, "Annotation_Boxes.xlsx")

meta_csv = "/home/maling/fanqiliang/data/Duke-Breast-Cancer-MRI/manifest-1607053360376/metadata.csv"

# Subject ID <=> Patient ID

import pandas as pd

annotation_dt = pd.read_excel(annotation_csv)
meta_dt = pd.read_csv(meta_csv)

annotation_dt.to_csv(os.path.join(save_root, "annotation.csv"))
print(f"annotation: {annotation_dt.columns.values}")
print(f"meta: {meta_dt.columns.values}")


import SimpleITK as sitk
from PIL import Image
import numpy as np
root = "/home/maling/fanqiliang/data/Duke-Breast-Cancer-MRI/manifest-1607053360376"
from glob import glob
import cv2

for i, row in meta_dt.iterrows():
    save_data_root = os.path.join(save_root, f"{i}", "data")
    save_label_root = os.path.join(save_root, f"{i}", "label")

    if not os.path.exists(save_data_root):
        os.makedirs(save_data_root)

    if not os.path.exists(save_label_root):
        os.makedirs(save_label_root)
    map_id = row["Subject ID"]
    _meta = annotation_dt[annotation_dt["Patient ID"] == map_id]
    print(map_id, "annotation num: ",len(_meta))
    draw = True   # no annotations
    try:
        start_row, end_row = int(_meta['Start Row'][0]), int(_meta['End Row'][0])
        start_column, end_column = int(_meta["Start Column"][0]), int(_meta['End Column'][0])
        start_slice, end_slice = int(_meta["Start Slice"][0]), int(_meta['End Slice'][0])
    except:
        draw = False

    path = str(row["File Location"]).lstrip(".\\").replace("\\", "/")
    path = os.path.join(root, path)
    slices = glob(os.path.join(path, "*.dcm"))
    slices.sort()
    for idx, dcm in enumerate(slices):
        dst_dcm = os.path.join(save_data_root, os.path.basename(dcm))
        dst_label = os.path.join(save_label_root, os.path.basename(dcm).replace(".dcm", ".jpg"))
        idx = idx + 1  # start from 1
        shutil.copyfile(dcm, dst_dcm)

        img = sitk.ReadImage(dcm)
        arr = sitk.GetArrayFromImage(img)[0]
        arr = np.asarray(Image.fromarray(arr).convert("RGB"))
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
        if draw and start_slice <= idx <= end_slice:
            cv2.rectangle(arr, (start_column, start_row), (end_column, end_row), color=(0, 0, 255), thickness=5)
            # cv2.rectangle(arr, (start_row, start_column), (end_row, end_column), color=(0, 0, 255), thickness=5)
        cv2.imwrite(dst_label, arr)
