#!/bin/bash

python train.py \
--weights '' \
--cfg 'yolov5s.yaml' \
--train_data '../medium_dataset/images/train' \
--test_data '../medium_dataset/images/validation'\
--hyp 'train_param.yaml' \
--epochs 100 \
--batch-size 16 \
--img-size 512 \
--adam \
--name 'experiment0' \
--project '/content/drive/MyDrive/yoloV5/train'
