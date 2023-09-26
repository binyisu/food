import os
from .meta_voc import register_meta_voc
from .voc_coco import register_voc_coco_test, register_voc_coco_train
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- COCO -------- #
def register_all_voc_coco(root="datasets"):

    SPLITS_TEST = [
        # VOC_COCO_openset
        ("voc_coco_test", "voc_coco", "instances_val2017"),# voc_coco测试数据集

    ]

    for name, dirname, split in SPLITS_TEST:
        year = 2007 if "2007" in name else 2012
        register_voc_coco_test(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    SPLITS_TRAIN = [("voc_2007_train1", "voc_coco", "voc07train"),# voc_coco训练base数据集
               ("voc_2012_trainval1", "voc_coco", "voc12trainval"), ]
    for prefix in ["all"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(20):
                seed = "_seed{}".format(seed)
                name = "voc_coco_trainval_{}_{}shot{}".format(
                    prefix, shot, seed
                )
                dirname = "voc_coco"
                SPLITS_TRAIN.append(
                    (name, dirname, name)
                )
    for name, dirname, split in SPLITS_TRAIN:
        year = 2007 if "2007" in name else 2012
        register_voc_coco_train(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),#voc2007base训练1
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),#voc2012base训练1
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),#voc2007base测试1
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        # ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        # ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        # ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "all_known_unknown_1", 1),#voc2007开集测试1
        ("voc_2007_test_all2", "VOC2007", "test", "all_known_unknown_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "all_known_unknown_3", 3)
    ]

# similar to the above, continue to register few-shot training data
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10, 30]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        # _seed1
                        seed = "_seed{}".format(seed)
                        # voc_2007_trainval_all1_1shot_seed1
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        # VOC2007
                        dirname = "VOC{}".format(year)
                        # base_novel_1
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, name, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

register_all_voc_coco()
register_all_voc()