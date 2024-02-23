# HSIC-based Moving WeightAveraging for Few-Shot Open-Set Object Detection

# News
[2024-01-18]：This is the official PyTorch implementation of Foodv1 and Foodv2.


[**Paper**](https://www.researchgate.net/publication/373451611_HSIC-based_Moving_Weight_Averaging_for_Few-Shot_Open-Set_Object_Detection)         [**Code**](https://github.com/binyisu/food)

![image](https://github.com/binyisu/food/blob/main/food.png)

## Setup

The code is based on detectron2 v0.3

- ### **Installation**

```
conda create -n Food python=3.8 -y
conda activate Food
```

- **Prepare datasets**

You should download：

- train and val set of COCO2017

- trainval and test set of VOC2007、VOC2012

following the structure described below：

```
datasets/
  coco/
  VOC20{07,12}/
```

In coco：

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
```

In  VOC20{07,12}：

```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
  JPEGImages/
```

Then we generate all datasets for FOOD:

```
bash prepare_food_voc_coco.sh
```

## Training and Evaluation

#### VOC-COCO dataset settings:

```
bash run_voc_coco_AR.sh
```

#### VOC10-5-5 dataset settings:

```
bash run_voc_AR.sh
```

### Citation

If you find this repo useful, please consider citing our paper:

```
@inproceedings{foodv2,
  title={HSIC-based Moving Weight Averaging for Few-Shot Open-Set Object Detection},
  author={Binyi Su, Hua Zhang, and Zhong Zhou},
  booktitle={Proceedings of the31st ACM International Conference on Multimedia (MM 23)},
  page={5358--5369},
  year={2023},
  doi={https://doi.org/10.1145/3581783.3611850}
}

@ARTICLE{10438382,
  author={Su, Binyi and Zhang, Hua and Li, Jingzhi and Zhou, Zhong},
  journal={IEEE Transactions on Image Processing}, 
  title={Toward Generalized Few-Shot Open-Set Object Detection}, 
  year={2024},
  volume={33},
  number={},
  pages={1389-1402},
  doi={10.1109/TIP.2024.3364495}}
```


