import SimpleITK as sitk
import cv2
import pandas as pd
import numpy as np
from typing import List
import os
from glob import glob
from PIL import Image
import json
from tqdm import tqdm
sitk.ProcessObject_GlobalWarningDisplayOff()


def parse_label(v: str) -> List:
    if "none" in v:
        return []
    else:
        b = v.split("opacity")
        b = [list(map(lambda x: float(x), i.strip().split(" ")))[1:]
             for i in b if len(i)]
        res = []
        for vec in b:
            res.append([vec[1], vec[0], vec[3], vec[2]])
        return res


def generate_annotation(data_root: str, label_csv: str, img_save_dir: str, save_path: str = None):
    dt = pd.read_csv(label_csv)

    annotation_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "target",
            }
        ]
    }

    ids: np.ndarray = dt["id"].apply(lambda v: v.replace("_image", "")).values
    labels: np.ndarray = dt["label"].values
    study_instances: np.ndarray = dt['StudyInstanceUID'].values

    annotation_idx = 0
    for i in tqdm(range(len(ids))):
        _id = ids[i]
        _bboxs = parse_label(labels[i])
        _study_instance = study_instances[i]

        img_path = glob(os.path.join(
            data_root, f"{_study_instance}", "*", f"{_id}.dcm"))[0]
        img: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(
            img_path, imageIO="GDCMImageIO")).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = np.asarray(Image.fromarray(img[0]).convert('RGB'))
        height = img.shape[0]
        width = img.shape[1]
        filename = f"{_study_instance}-{_id}.jpeg"
        cv2.imwrite(os.path.join(img_save_dir, filename), img)

        image_obj = {
            "id": i,
            "width": width,
            "height": height,
            "file_name": filename
        }
        annotation_json["images"].append(image_obj)

        for y1, x1, y2, x2 in _bboxs:
            annotation = {
                "id": annotation_idx,
                "image_id": f"{i}",
                "bbox": [x1, y1, x2-x1, y2-y1],
                "iscrowd": 1
            }
            annotation_idx += 1
            annotation_json["annotations"].append(annotation)
    save_path = save_path if save_path is not None else "."
    json.dump(annotation_json, open(os.path.join(
        f"{save_path}", "annotation.json"), "w"), indent=4)


def main():
    pass


if __name__ == "__main__":
    data_root = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train"
    label_csv = "/home/maling/fanqiliang/data/kaggle/siim-covid19-detection/train_image_level.csv"

    generate_annotation(data_root, label_csv, img_save_dir="/home/maling/fanqiliang/data/coco_covid19/imgs",
                        save_path="/home/maling/fanqiliang/data/coco_covid19")
