import torch
import torch.nn as nn
from utils.general import *


# class CSP(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#
#     def __init__(self, c1, c2, n=3, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(CSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cbl1 = CBL(c1, c_, 1, 1)
#         self.cbl2 = CBL(c1, c_, 1, 1)
#         self.cbl3 = CBL(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[ResUnit(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cbl3(torch.cat((self.m(self.cbl1(x)), self.cbl2(x)), dim=1))
#
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = CBL(c1, c_, 1, 1)
        self.cv2 = CBL(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.upsample



class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.CBL1 = CBL(c1, c_, 1, 1)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.CBL2 = CBL(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[ResUnit(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv2(self.m(self.CBL1(x)))
        y2 = self.cv1(x)
        return self.CBL2(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP_2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP_2, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.CBL1 = CBL(c1, c_, 1, 1)
        self.m = nn.Sequential(*[CBL(c_, c_, 1, 1) for _ in range(n * 2)])
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.CBL2 = CBL(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, x):
        y1 = self.cv2(self.m(self.CBL1(x)))
        y2 = self.cv1(x)
        return self.CBL2(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class CBL(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,padding=None,g=1,activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k,padding),groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if activation is True else(activation if isinstance(activation,nn.Module) else nn.Identity())
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))



class ResUnit(nn.Module):
    # standard bottleneck -> reduce the calculation
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5): # ch_in, ch_out, shortcut, groups, expansion
        super(ResUnit, self).__init__()
        c_ = int(c2*e) #hidden channels
        self.conv1 = CBL(c1,c_,k =1, s=1)
        self.conv2 = CBL(c_,c2,k=3,s=1,g=g)
        self.add = shortcut and c1==c2 #if shortcut and input channel == output channel
    def forward(self,x):
        return x+self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas): # boxes is the anchors deltas is the output value from the network

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes