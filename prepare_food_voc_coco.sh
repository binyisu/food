DATA_DIR=datasets/voc_coco
COCO_DIR=datasets/coco
VOC07_DIR=datasets/VOC2007
VOC12_DIR=datasets/VOC2012

# make neccesary dirs
rm $DATA_DIR -rf
echo "make dirs"
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/Annotations
# mkdir -p DATA_DIR/JPEGImages
mkdir -p $DATA_DIR/ImageSets
mkdir -p $DATA_DIR/ImageSets/Main

# cp data
# make use you have $COCO_DIR, VOC07_DIR and VOC12_DIR
echo "copy coco images"
cp $COCO_DIR/train2017 $DATA_DIR/JPEGImages -r
cp $COCO_DIR/val2017/* $DATA_DIR/JPEGImages/

echo "convert coco annotation to voc"
python datasets/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_train2017.json
python datasets/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_val2017.json



echo "copy voc images"
cp $VOC07_DIR/JPEGImages/* $DATA_DIR/JPEGImages/
cp $VOC12_DIR/JPEGImages/* $DATA_DIR/JPEGImages/

echo "copy voc annotation"
cp $VOC07_DIR/Annotations/* $DATA_DIR/Annotations/
cp $VOC12_DIR/Annotations/* $DATA_DIR/Annotations/

echo "copy voc imagesets"
cp $VOC07_DIR/ImageSets/Main/train.txt $DATA_DIR/ImageSets/Main/voc07train.txt
cp $VOC07_DIR/ImageSets/Main/val.txt $DATA_DIR/ImageSets/Main/voc07val.txt
cp $VOC07_DIR/ImageSets/Main/test.txt $DATA_DIR/ImageSets/Main/voc07test.txt
cp $VOC12_DIR/ImageSets/Main/trainval.txt $DATA_DIR/ImageSets/Main/voc12trainval.txt

echo "generate voc_coco_val imagesets"
cat $DATA_DIR/ImageSets/Main/voc07val.txt > $DATA_DIR/ImageSets/Main/voc_coco_val.txt
cat $DATA_DIR/ImageSets/Main/instances_val2017.txt >> $DATA_DIR/ImageSets/Main/voc_coco_val.txt

echo "generate few-shot imagesets"
python prepare_voc_few_shot.py
python prepare_voc_coco_few_shot.py