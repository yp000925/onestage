from model import yolov5
import torch, torchvision
from utils.datasets import create_dataloader_modified
import yaml
import os
import argparse
import torch.optim as optim
import time
from tqdm import tqdm
import logging
from model import *
from utils.general import *
from torch.utils.tensorboard import SummaryWriter
from utils.plot_utils import *

ckpt_path = '/Users/zhangyunping/PycharmProjects/onestage/train/bs32/best.pt'

hyp = '/Users/zhangyunping/PycharmProjects/onestage/train_param.yaml'

with open(hyp) as f:
    hyp = yaml.load(f)

model = yolov5.Model(hyp)
ckpt = torch.load(ckpt_path,map_location=torch.device('cpu'))


if torch.cuda.is_available():
    model = model.to('cuda')
    model = torch.nn.DataParallel(model).cuda()
    model.device = torch.device('cuda')
else:
    model = torch.nn.DataParallel(model)
    model.device = torch.device('cpu')

state_dict = ckpt['param']
model.load_state_dict(state_dict)

source = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
batch_size = 16
img_size = 512
dataloader = create_dataloader_modified(source, img_size, batch_size)[0]

model.eval()
nc = 256  # number of classes

mat = ConfusionMatrix(nc=256, conf=0, iou_thres=0.6)

with torch.no_grad():
    for batch_i, (img, targets, paths) in enumerate(tqdm(dataloader)):
        # targets in the format [batch_idx, class_id, x,y,w,h]
        img = img.to(model.device )
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        targets = targets.to(model.device )
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(model.device )  # to pixels

        out, train_out = model(img)  # inference and training outputs

        # # if would like to use one_hot for output
        # out = pred_label_onehot(out)
        # out = non_max_suppression(out)
        # out = post_nms(out,0.45)# list of anchors with [xyxy, conf, cls]

        # # if would like to use depthmap as the class directly
        out = nms_modified(out, obj_thre=0.8, iou_thres=0.5, nc=256)  # list of anchors with [xyxy, conf, cls]
        # # 因为用了torch自带的nms所以变成了xyxy
        #
        # # plot ----------------------------------------------------------------------------------------------------------------
        # # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        # plot_images_modified(img, targets, paths, fname='check.jpg', names=None)
        # plot_images_modified(img, output_to_target(out), paths, fname='check_pred2.jpg', names=None)
        # break

# # update confusion matrix ----------------------------------------------------------------------------------------------
#       for batch_idx in range(len(out)):
#           labels = targets[targets[:,0].int()==batch_idx][:,1::] # class, x,y,w,h
#           detections = out[batch_idx] # x,y,x,y,conf,cls
#           detections[:,5] = detections[:,5].int()
#           labels[:,1::] = xywh2xyxy(labels[:,1::])# class, x,y,x,y
#           mat.process_batch(detections,labels)

# # Calculate accuracy      ----------------------------------------------------------------------------------------------
#   mtx = mat.matrix
#   thred = 10 # take prediction within this range as acceptable
#   correct_match = 0
#   total_num = np.sum(mtx[:,0:nc])
#   for gt_cls in range(mtx.shape[1]-1):
#       correct_match += np.sum(mtx[max(0,gt_cls-thred):min(gt_cls+thred,mtx.shape[0]-1), gt_cls])
#   accuracy = correct_match/total_num
#   res = 0.02/256
#   print("The total accuracy for boundary {:f}mm is {:f}%".format(thred*res*1000,accuracy*100))
