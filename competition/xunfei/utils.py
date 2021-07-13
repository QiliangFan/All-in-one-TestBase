import pandas as pd
from torch.utils.data import Dataset
from typing import Sequence
from glob import glob
import os
import json
import xmltodict


def load_train_data(data_root: str):
    domains = glob(os.path.join(data_root, "**", "domain*"), recursive=True)
    imgs = []
    bounding_box = []
    for domain in domains:
        xml = os.path.join(domain, "XML")
        annotations = glob(os.path.join(xml, "*.xml"))
        for an in annotations:
            with open(an, "r") as fp:
                xml_dict = xmltodict.parse(fp.read())
                img_path = os.path.join(domain, xml_dict['annotation']['filename'])
                objects = xml_dict['annotation']['object']
                bboxs = {}
                if "name" in objects:
                    y_min, x_min, y_max, x_max = float(objects['bndbox']['ymin']), float(objects['bndbox']['xmin']), float(objects['bndbox']['ymax']), float(objects['bndbox']['xmax'])
                    if objects['name'] in bboxs:
                        bboxs[objects['name']].append([y_min, x_min, y_max-y_min, x_max-x_min])
                    else:
                        bboxs[objects['name']] = [[y_min, x_min, y_max-y_min, x_max-x_min]]
                else:
                    for o in objects:
                        y_min, x_min, y_max, x_max = float(o['bndbox']['ymin']), float(o['bndbox']['xmin']), float(o['bndbox']['ymax']), float(o['bndbox']['xmax'])
                        if o['name'] in bboxs:
                            bboxs[o['name']].append([y_min, x_min, y_max-y_min, x_max-x_min])
                        else:
                            bboxs[o['name']] = [[y_min, x_min, y_max-y_min, x_max-x_min]]
                imgs.append(img_path)
                bounding_box.append(bboxs)
    return imgs, bounding_box
