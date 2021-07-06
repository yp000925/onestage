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

logger = logging.getLogger(__name__)


def train_epoch(model, optimizer, dataloader,epoch,freeze = [], tb_writer=None):
    model.train()
    # Freeze
    # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    nb = len(train_dataloader)
    t0 = time.time()
    mloss = torch.zeros(4, device=model.device)  # mean losses
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=nb)
    logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'depth', 'total', 'labels', 'img_size'))

    optimizer.zero_grad()
    for i, (imgs, targets, paths) in pbar:  # batch ------------------
        ni = i + nb * epoch
        imgs = imgs.to(model.device, non_blocking=True).float() / 255.0
        preds, loss, loss_items = model((imgs,targets))
        # = model.compute_loss(preds, targets)
        loss.backward()
        optimizer.step()

        # print
        mloss = (mloss * i + loss_items) / (i + 1)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        info = ('%10s' * 2 + '%10.4g' * 6) % (
            '%g' % (epoch), mem, *mloss, targets.shape[0], imgs.shape[-1])
        pbar.set_description(info)

    return mloss


def eval_epoch(model, dataloader):
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar,total=len(dataloader))
    mloss = torch.zeros(4, device=model.device)
    logger.info('\n Evaluation========================================')
    logger.info(('\n' + '%10s' * 4) % ('box', 'obj', 'depth', 'total'))
    model.eval()
    mat = ConfusionMatrix(nc=256, conf=0, iou_thres=0.6)  # conf should change accordingly 0.8 for depthmap

    with torch.no_grad():

        for i,(imgs,targets,paths) in pbar:
            imgs = imgs.to(model.device).float()/255.0
            out, train_out = model(imgs)
            # numerical eval
            numerical_eval(mat,out,targets)

            loss, loss_items = model.module.compute_loss(train_out,targets)
            mloss = (mloss*i+loss_items)/(i+1)
            info = ('%10.4g' * 4) % (*mloss,)
            pbar.set_description(info)

        mtx = mat.matrix
        thred = 10  # take prediction within this range as acceptable
        correct_match = 0
        nc = 256
        total_num = np.sum(mtx[:, 0:nc])
        for gt_cls in range(mtx.shape[1] - 1):
            correct_match += np.sum(mtx[max(0, gt_cls - thred):min(gt_cls + thred, mtx.shape[0] - 1), gt_cls])
        accuracy = correct_match / total_num
        res = 0.02 / 256

        logger.info("The total accuracy for boundary {:f}um is {:f}%".format(thred * res * 1000, accuracy * 100))

    return mloss, accuracy

def numerical_eval(mat,outs,targets):
    preds = post_nms(outs,0.45)
    for batch_idx in range(len(preds)): #batch_idx 指的是图片在batch里面的idx 即一张一张图片过
        labels = targets[targets[:, 0].int() == batch_idx][:, 1::]  # class, x,y,w,h
        detections = preds[batch_idx]  # x,y,x,y,conf,cls
        detections[:, 5] = detections[:, 5].int()
        labels[:, 1::] = xywh2xyxy(labels[:, 1::])  # class, x,y,x,y
        mat.process_batch(detections, labels)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='last.pt', help='load weights path')
    parser.add_argument('--train_data', type=str, default='images/train_data', help='training data path')
    parser.add_argument('--eval_data', type=str, default='images/eval_data', help='evaluation data path')
    parser.add_argument('--test_data', type=str, default='images/test_data', help='test data path')
    parser.add_argument('--hyp', type=str, default='train_param.yaml', help='hyperparameter path')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=512, help='image sizes')
    parser.add_argument('--project', default='train', help='save to project/name')
    parser.add_argument('--experiment',default='experiment',help ='experiment name')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    opt = parser.parse_args()
    # opt.train_data = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
    # opt.test_data = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'

    save_dir = os.path.join(opt.project, opt.experiment)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tb_writer = SummaryWriter(save_dir)

    last_path = os.path.join(opt.project, opt.experiment, 'last.pt')
    best_path = os.path.join(opt.project, opt.experiment, 'best.pt')

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    model_name = hyp['model_name']
    model = eval(model_name).Model(hyp)
    if torch.cuda.is_available():
        model = model.to('cuda')
        model = torch.nn.DataParallel(model).cuda()
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')


    train_dataloader, dataset = create_dataloader_modified(opt.train_data, opt.img_size, opt.batch_size)
    test_dataloader, dataset = create_dataloader_modified(opt.test_data, opt.img_size, opt.batch_size)

    # Optimizer
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'],
                               betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # resume
    start_epoch = 0
    best_acc = 0.0
    scheduler.last_epoch = start_epoch - 1

    end_epoch = start_epoch + opt.epochs

    # if pretrainded:


    # train loop
    for epoch in range(start_epoch, end_epoch, 1):
        train_loss = train_epoch(model, optimizer, train_dataloader, epoch)
        eval_loss, accuracy = eval_epoch(model,test_dataloader)

        # Log
        current_lr = optimizer.param_groups[0]['lr']
        tags = ['train/box_loss', 'train/obj_loss', 'train/depth_loss',  # train loss
                'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                'accuracy','lr']  # params
        for x, tag in zip(list(train_loss[:-1]) + list(eval_loss[:-1]) + list([accuracy])+list([current_lr]), tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

        #save the last

        ckpt = {
            'param': model.state_dict(),
            'model_name': model_name,
            'last_epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpt,last_path)
        logger.info("Epoch {:d} saved".format(epoch))

        # update the best
        if accuracy > best_acc:
            best_acc = accuracy
        # save the best
        if best_acc == accuracy:
            torch.save(ckpt, best_path)
            logger.info("Epoch {:d} is the best currently".format(epoch))


