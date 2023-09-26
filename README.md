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
### Citation

If you find this repo useful, please consider citing our paper:

```
@inproceedings{zhong2022regionclip,
  title={HSIC-based Moving WeightAveraging for Few-Shot Open-Set Object Detection},
  author={Binyi Su, Hua Zhang, and Zhong Zhou},
  booktitle={Proceedings of the31st ACM International Conference on Multimedia (MM 23)},
  year={2023}
  doi={https://doi.org/10.1145/3581783.3611850}
}
```


