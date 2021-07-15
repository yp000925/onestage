from model.common import *

class Backbone(nn.Module):

    def __init__(self,ch = 3):# input channel
        super(Backbone, self).__init__()
        self.ch = ch
        self.focus = Focus(ch,32,3,1,p=None,g=1) # s/2
        self.CBL1 = CBL(32,64,3,2, padding=None, g=1, activation=True) # s/4
        self.CSP1_1 = BottleneckCSP(64, 64, n=1, shortcut=True, g=1, e=0.5)
        self.CBL2 = CBL(64,128,3,2,padding=None,g=1,activation=True) #s/8
        self.CSP2_3 = BottleneckCSP(128,128,n=3,shortcut=True,g=1,e=0.5) #output feature to Neck
        self.CBL3 = CBL(128,256,3,2,padding=None,g=1,activation=True) # s/16
        self.CSP3_3 = BottleneckCSP(256,256,n=3,shortcut=True,g=1,e=0.5) #output feature to Neck
        self.CBL4 = CBL(256,512,3,2,padding=None,g=1,activation=True) # s/32
        self.SPP = SPP(512,512,(5,9,13))
        self.CSP4_1 = BottleneckCSP_2(512,512,n=1,shortcut=True,g=1,e=0.5)
        self.CBL5 = CBL(512,256,k=1,s=1,padding=None,g=1,activation=True) #output feature to Neck

    def forward(self,x):
        focus = self.focus(x)
        x = self.CBL1(focus)
        x = self.CSP1_1(x)
        x =self.CBL2(x)
        P0 = self.CSP2_3(x)
        x = self.CBL3(P0)
        P1 = self.CSP3_3(x)
        x = self.CBL4(P1)
        x = self.SPP(x)
        x = self.CSP4_1(x)
        P2 = self.CBL5(x)
        return [P0, P1, P2]

class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.upsample1 = nn.Upsample(None,2,'nearest')
        self.concat = Concat(dimension=1)
        self.CSP5_1 = BottleneckCSP_2(256*2,256,n=1,shortcut=True,g=1,e=0.25)
        self.CBL6 = CBL(256,128,1,1,padding=None,g=1,activation=True)
        self.upsample2 = nn.Upsample(None,2,'nearest')
        self.CSP6_1 = BottleneckCSP_2(128 * 2, 128, n=1, shortcut=True, g=1, e=0.25)
        self.concat2 = Concat(dimension=1)
        self.CBL7 = CBL(128,128,3,2,padding=None,g=1,activation=True)
        self.concat3 = Concat(dimension=1)
        self.CSP7_1 = BottleneckCSP_2(128 * 2, 256, n=1, shortcut=True, g=1, e=0.5)
        self.CBL8 = CBL(256,256,3,2,padding=None,g=1,activation=True)
        self.concat4 = Concat(dimension=1)
        self.CSP8_1 = BottleneckCSP_2(256 * 2, 512, n=1, shortcut=True, g=1, e=0.5)

    def forward(self,x):
        P0 = x[0]
        P1 = x[1]
        P2 = x[2]
        upsample1 = self.upsample1(P2)
        l1 = self.concat([upsample1,P1])
        l2 = self.CSP5_1(l1)
        l3 = self.CBL6(l2)
        l4 = self.upsample2(l3)
        l5 = self.concat2([l4,P0])
        l6 = self.CSP6_1(l5) # output to detection layer torch.Size([-1, 128, 64, 64]) s/8
        l7 = self.CBL7(l6)
        l8 = self.concat3([l7,l3])
        l9 = self.CSP7_1(l8) # output to detection layer torch.Size([-1, 256, 32, 32]) s/16
        l10 = self.CBL8(l9)
        l11 = self.concat4([l10, P2])
        l12 = self.CSP8_1(l11) # output to detection layer torch.Size([-1, 512, 16, 16]) s/32
        return [l6,l9,l12]



class DetectLinearHead(nn.Module):
     # strides computed during build

    def __init__(self, ch=(128,256,512), inplace=True):  # detection layer
        super(DetectLinearHead, self).__init__()
        self.no = 6  # number of outputs per anchor
        self.nl = 3  # number of detection layers
        self.na = 3  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl): #for each layer output
            x[i] = self.m[i](x[i])  # conv shape (-1,255,ny,nx)
            bs, _, ny, nx = x[i].shape  # x(bs,18,ny,nx) to x(bs,#anchor,ny,nx,6)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # here we do not perform any activation, just the directly output from the convolution
        return x  # x-> the reshaped output for each layer


class Model(nn.Module):
    def __init__(self,hyp):
        super(Model, self).__init__()
        anchors = np.array([[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]])

        self.no = 6
        self.nl = 3
        self.na = 3
        self.stride = torch.tensor([8, 16, 32])
        anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        a = anchors/self.stride.view(-1, 1, 1)
        # this buffer needed to be stored in the state_dict for further inference use
        self.register_buffer('scaled_anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', anchors.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.head = DetectLinearHead(ch=(128,256,512))
        self.backbone = Backbone()
        self.neck = Neck()
        self.grid = [torch.zeros(1)] * self.nl
        self.boxgain = hyp['box']
        self.objgain = hyp['obj']
        self.depthgain = hyp['depth']

    def forward(self,x):
        if isinstance(x,tuple):
            imgs, targets = x
        else:
            imgs = x

        [p0,p1,p2] = self.backbone(imgs)
        [o0,o1,o2] = self.neck([p0,p1,p2])
        out = self.head([o0,o1,o2])
        assert len(out) == self.nl

        if not self.training:
            z = []
            for i in range(self.nl):# inference
                bs, _, ny, nx, _ = out[i].shape
                if self.grid[i].shape[2:4] != out[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(out[i].device)
                # y = x[i].sigmoid()
                y = out[i] # shape (bs,#anchor,ny,nx,6)
                # print(y.shape)
                xy = (y[..., 0:2].sigmoid() * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4].sigmoid() * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                y = torch.cat((xy, wh, y[..., 4:5].sigmoid(), y[..., 5:]), -1)  # xy,wh, objectiveness, pred_depth
                z.append(y.view(bs, -1, self.no))
            return (torch.cat(z, 1), out)

        else:
            loss, loss_items = self.compute_loss(out,targets)
            return out, loss, loss_items


    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _build_targets(self, preds, targets):
            # Build targets for compute_loss(), input targets(image_idx,depth 0-255,x,y,w,h)
            na, nt = self.na, targets.shape[0]  # number of anchors, targets
            tcls, tbox, indices, anch = [], [], [], []
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
            ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1,nt)  # same as .repeat_interleave(nt)
            targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

            g = 0.5  # bias
            off = torch.tensor([[0, 0],
                                [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                                ], device=targets.device).float() * g  # offsets

            for i in range(self.nl): # 对每一层都要创建targets
                anchors = self.scaled_anchors[i] # shape(na,2)
                gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain -> feature map size
                # Match targets to anchors
                t = targets * gain
                if nt:
                    # Matches
                    r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                    anchor_t = 4.0 # anchor ratio threshold


                    j = torch.max(r, 1. / r).max(2)[0] < anchor_t  # compare 只保留和gt的shape满足一定比率的anchors，但并没有选最大值？
                    # 并没有保证每个bbox gt只对应一个anchor来做回归
                    # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                    t = t[j]  # filter

                    # Offsets
                    gxy = t[:, 2:4]  # grid xy
                    gxi = gain[[2, 3]] - gxy  # inverse
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                    j = torch.stack((torch.ones_like(j), j, k, l, m))
                    t = t.repeat((5, 1, 1))[j]
                    offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                else:
                    t = targets[0]
                    offsets = 0

                # Define
                b, c = t[:, :2].long().T  # imageidx in batch, class
                gxy = t[:, 2:4]  # grid xy
                gwh = t[:, 4:6]  # grid wh
                gij = (gxy - offsets).long()
                gi, gj = gij.T  # grid xy indices

                # Append
                a = t[:, 6].long()  # anchor indices
                indices.append(
                    (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
                tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
                anch.append(anchors[a])  # anchors
                tcls.append(c)  # class

            return tcls, tbox, indices, anch

    def compute_loss(self, preds, targets):
        device = targets.device
        BCEobj = nn.BCEWithLogitsLoss()
        MSEdepth = nn.MSELoss()

        ldepth, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self._build_targets(preds, targets)  # targets

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression 这里和inference 保持一致就好
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # obji = self.BCEobj(pi[..., 4], tobj)
                # lobj += obji * self.balance[i]  # obj loss
                lobj += BCEobj(pi[..., 4], tobj)

                # depth loss
                t = tcls[i]/255.0 # change to 0-1 range
                ldepth += MSEdepth(ps[:, 5], t)

            # if self.autobalance:
            #     self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        lbox *= self.boxgain
        lobj *= self.objgain
        ldepth *= self.depthgain
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + ldepth
        return loss * bs, torch.cat((lbox, lobj, ldepth, loss)).detach()

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    # m = torch.ones((2, 3, 512, 512), dtype=torch.float64)
    # model = Backbone().to(torch.float64)
    from utils.datasets import create_dataloader_modified
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    source = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
    img_size = 512
    batch_size = 2

    dataloader = create_dataloader_modified(source, img_size, batch_size)[0]

    # project = '/content/drive/MyDrive/yoloV5/train/exp3'
    device = torch.device('cpu')
    for batch_i, (img, targets, paths) in enumerate(dataloader):
        img = img.float()
        img /= 255.0
        break

    # m = torch.ones((2, 3, 512, 512))
    # model = Backbone()
    # c = model(m)
    # model2  = Neck()
    # c2 = model2(c)
    hyp={}
    # hyp['no'] = 6
    # hyp['na'] = 3
    # hyp['nl'] = 3
    hyp['box'] = 1
    hyp['obj'] = 1
    hyp['depth'] = 1
    # anchors = np.array([[10,13, 16,30, 33,23], [30,61, 62,45, 59,119], [116,90, 156,198, 373,326]])
    anchors = np.array([[68.411, 67.417,80.912, 80.904,72.071, 113.98],
           [115.41, 73.2,92.348, 92.396,103.63, 103.64],
           [114.36, 114.33,124.89, 124.9,137.04, 137.08]])
    model = Model(hyp)

    preds = model((img,targets))

    # model._build_targets(preds,targets)
    # model.compute_loss(preds, targets)




