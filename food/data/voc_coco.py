import io
import os
import contextlib
import xml.etree.ElementTree as ET
import numpy as np
from pycocotools.coco import COCO
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_voc_instances
from typing import List, Tuple, Union



VOC_COCO_CATEGORIES = [
    # VOC
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    # COCO-20-40
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    # COCO-40-60
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    # COCO-60-80
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    # Unknown
    "unknown",
]

VOC_COCO_TRAIN_CATEGORIES = [
    # VOC
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    # COCO-20-40
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator",]


def register_voc_coco_test(name, dirname, split, year):
    class_names = VOC_COCO_CATEGORIES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_voc_coco_train(name, dirname, split, year):
    class_names = VOC_COCO_TRAIN_CATEGORIES
    if "shot" in split:
        func = load_filtered_voc_coco_instances
    else:
        func = load_voc_instances
    DatasetCatalog.register(
        name, lambda: func(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def load_filtered_voc_coco_instances(
    dirname: str, split: str, classnames: Union[List[str], Tuple[str, ...]]
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    name = split
    fileids = {}
    split_dir = os.path.join("datasets", "voccocosplit")
    shot = name.split("_")[-2].split("shot")[0]
    seed = int(name.split("_seed")[-1])
    split_dir = os.path.join(split_dir, "seed{}".format(seed))
    for cls in classnames:
        with PathManager.open(
                os.path.join(
                    split_dir, "box_{}shot_{}_train.txt".format(shot, cls)
                )
        ) as f:
            fileids_ = np.loadtxt(f, dtype=np.str).tolist()
            if isinstance(fileids_, str):
                fileids_ = [fileids_]
            fileids_ = [
                fid.split("/")[-1].split(".jpg")[0] for fid in fileids_
            ]
            fileids[cls] = fileids_

    dicts = []
    for cls, fileids_ in fileids.items():
        dicts_ = []
        for fileid in fileids_:
            dirname = os.path.join("datasets", "voc_coco")
            anno_file = os.path.join(
                dirname, "Annotations", fileid + ".xml"
            )
            jpeg_file = os.path.join(
                dirname, "JPEGImages", fileid + ".jpg"
            )

            tree = ET.parse(anno_file)

            for obj in tree.findall("object"):
                r = {
                    "file_name": jpeg_file,
                    "image_id": fileid,
                    "height": int(tree.findall("./size/height")[0].text),
                    "width": int(tree.findall("./size/width")[0].text),
                }
                cls_ = obj.find("name").text
                if cls != cls_:
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances = [
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                ]
                r["annotations"] = instances
                dicts_.append(r)
        if len(dicts_) > int(shot):
            dicts_ = np.random.choice(dicts_, int(shot), replace=False)
        dicts.extend(dicts_)
    return dicts
