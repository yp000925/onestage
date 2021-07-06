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
