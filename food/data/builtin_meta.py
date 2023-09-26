# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json

PASCAL_VOC_ALL_CATEGORIES_21 = {

    1: ["aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat","chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        # Unknown
        "unknown",
    ],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "aeroplane", "bottle", "cow", "horse", "sofa", "unknown",
    ],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor",
        "boat", "cat", "motorbike", "sheep", "sofa", "unknown",
    ],
}

# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: ["aeroplane", "bicycle", "boat", "bottle", "car",
        "cat", "chair", "diningtable", "dog", "horse",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "bird", "bus", "cow", "motorbike", "sofa",
    ],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
        "person", "pottedplant", "sheep", "train", "tvmonitor",
        "aeroplane", "bottle", "cow", "horse", "sofa",
    ],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
        "horse", "person", "pottedplant", "train", "tvmonitor",
        "boat", "cat", "motorbike", "sheep", "sofa",
    ],
}


PASCAL_VOC_BASE_CATEGORIES = {
    1: ["aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow"
    ],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike",
    ],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog",
    ],
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ["diningtable", "dog", "horse", "motorbike", "person",],
    2: ["person", "pottedplant", "sheep", "train", "tvmonitor",],
    3: ["horse", "person", "pottedplant", "train", "tvmonitor",],
}

PASCAL_VOC_KNOWN_CATEGORIES = {
    1: ["aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",],
    2: ["bicycle", "bird", "boat", "bus", "car",
        "cat", "chair", "diningtable", "dog", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor",],
    3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
        "car", "chair", "cow", "diningtable", "dog", "horse", "person", "pottedplant", "train", "tvmonitor",],
}

def _get_voc_fewshot_instances_meta():
    ret = {
        "thing_classes_21": PASCAL_VOC_ALL_CATEGORIES_21,
        "thing_classes": PASCAL_VOC_ALL_CATEGORIES,
        "known_classes": PASCAL_VOC_KNOWN_CATEGORIES,
        "novel_classes": PASCAL_VOC_NOVEL_CATEGORIES,
        "base_classes": PASCAL_VOC_BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    # if dataset_name == "coco":
    #     return _get_coco_instances_meta()
    # elif dataset_name == "coco_fewshot":
    #     return _get_coco_fewshot_instances_meta()
    if dataset_name == "voc_fewshot":
        return _get_voc_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
