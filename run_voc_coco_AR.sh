#!/usr/bin/env bash

EXP_NAME=voc_coco
METHOD_NAME=HMWA_2
SAVE_DIR=output/${EXP_NAME}
IMAGENET_PRETRAIN=ImageNetPretrained/MSRA/R-50.pkl                           # <-- change it to you path
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth  # <-- change it to you path

#
# ------------------------------- Base Pre-train ---------------------------------- #
python main.py --num-gpus 1 --config-file configs/${EXP_NAME}/food_det_r50_voc_coco_base.yaml --opts OUTPUT_DIR ${SAVE_DIR}/food_r50_voc_coco_base \
 MODEL.WEIGHTS ${IMAGENET_PRETRAIN}
python tools/model_surgery.py --dataset voc_coco --method remove --src-path ${SAVE_DIR}/food_r50_voc_coco_base/model_final.pth \
 --save-dir ${SAVE_DIR}/food_r50_voc_coco_base

BASE_WEIGHT=${SAVE_DIR}/food_r50_voc_coco_base/model_reset_remove.pth
## ------------------------------ Novel Fine-tuning ------------------------------- #
## --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
for seed in 1 2 3 4 5 6 7 8 9 10
do
    for shot in 1 5 10 30 # if final, 10 -> 1 2 3 5 10 30
    do
        python tools/create_config.py --dataset voc_coco --config_root configs/voc_coco               \
            --shot ${shot} --seed ${seed} --setting 'gfsod'
        CONFIG_PATH=configs/voc_coco/food_gfsod_r50_novel_${shot}shot_seed${seed}.yaml
        OUTPUT_DIR=${SAVE_DIR}/${METHOD_NAME}/${shot}shot_seed${seed}
        python main.py --num-gpus 1 --config-file ${CONFIG_PATH}                            \
            --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                          \
                   TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}                                  \
                   SOLVER.IMS_PER_BATCH 4
        rm ${CONFIG_PATH}
        rm ${OUTPUT_DIR}/model_final.pth
    done
done

python tools/extract_results.py --res-dir ${SAVE_DIR}/${METHOD_NAME} --shot-list 1 5 10 30 # surmarize all results