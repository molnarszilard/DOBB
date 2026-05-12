# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from .block import DFL, Proto, ContrastiveHead, BNContrastiveHead
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "DOBB"


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)
    
class DOBB(Detect):
    """YOLOv8 DOBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, extra_shape=[1,2], dir_type="vector", descriptors_size=16,ch=()):
        """Initialize DOBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne #3 #ne  # number of extra parameters
        self.extra_shape = extra_shape[0]*extra_shape[1]
        self.dir_type = dir_type
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        c5 = max(ch[0] // 4, self.extra_shape)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.extra_shape, 1)) for x in ch)
        self.descriptors_size = descriptors_size
        if self.descriptors_size:
            self.c6 = self.descriptors_size
            self.cv6 = nn.ModuleList(nn.Sequential(Conv(x, self.c6, 3), Conv(self.c6, self.c6, 3), nn.Conv2d(self.c6, self.c6, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if not isinstance(self.extra_shape, int):
            if len(self.extra_shape)>1:
                self.extra_shape=self.extra_shape[0]*self.extra_shape[1]
        bs = x[0].shape[0]  # batch size
        ### angleBB is the angle of the bounding box
        angleBB = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        ### NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angleBB = (angleBB.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4] 
        
        direction_ = torch.cat([self.cv5[i](x[i]).view(bs, self.extra_shape, -1) for i in range(self.nl)], -1)
        if self.dir_type in ["vector"]:
            direction_[:,0,:] = (direction_[:,0,:].sigmoid()-0.5)*2
            direction_[:,1,:] = (direction_[:,1,:].sigmoid()-0.5)*2
        if direction_.isnan().any():
            print("Head is nan") # Sometimes the model returns NaN during warmup. If this is NaN outside of warmup it is a problem.
        if self.descriptors_size:
            descriptors_repeated, descriptors = self.concat_descriptors(x)
        if not self.training:
            self.angleBB = angleBB
            self.direction_ = direction_
        x = self.detect(self, x)
        if self.training:
            if self.descriptors_size:
                return x, angleBB, direction_, descriptors
            else:
                return x, angleBB, direction_
        pred_direction_=direction_
        if self.descriptors_size:
            return torch.cat([x,angleBB, pred_direction_,descriptors_repeated], 1) if self.export else (torch.cat([x[0], angleBB, pred_direction_,descriptors_repeated], 1), (x[1], angleBB, direction_, descriptors_repeated))
        else:
            return torch.cat([x,angleBB, pred_direction_], 1) if self.export else (torch.cat([x[0], angleBB, pred_direction_], 1), (x[1], angleBB, direction_))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angleBB, anchors, dim=1)
    
    def concat_descriptors(self,x):
        bs = x[0].shape[0]
        ### Process the descriptors
        # Read the 3 layers of the descriptors
        descriptors_layer_0 = self.cv6[0](x[0])
        descriptors_layer_1 = self.cv6[1](x[1])
        descriptors_layer_2 = self.cv6[2](x[2])
        # Get their size, 0 should be the largest, but let's make sure that we know the max size
        desc_h_0,desc_w_0 = descriptors_layer_0.shape[-2:]
        desc_h_1,desc_w_1 = descriptors_layer_1.shape[-2:]
        desc_h_2,desc_w_2 = descriptors_layer_2.shape[-2:]
        # desc_h = max(desc_h_0,desc_h_1,desc_h_2)
        # desc_w = max(desc_w_0,desc_w_1,desc_w_2)
        # Resize the descriptors, so each of them have the same size (maximum), without interpolation, using the nearest value to fill in the gaps
        descriptors_layer_0 = torch.nn.functional.interpolate(descriptors_layer_0,size=(desc_h_0,desc_w_0),mode='nearest-exact').view(bs, self.c6, -1)
        descriptors_layer_1 = torch.nn.functional.interpolate(descriptors_layer_1,size=(desc_h_0,desc_w_0),mode='nearest-exact').view(bs, self.c6, -1)
        descriptors_layer_2 = torch.nn.functional.interpolate(descriptors_layer_2,size=(desc_h_0,desc_w_0),mode='nearest-exact').view(bs, self.c6, -1)
        # concatenate the descriptor vectors in each anchor grid point, therefore the vector length is 3*self.c6 (e.g., 16*3=48)
        descriptors = torch.cat([descriptors_layer_0,descriptors_layer_1,descriptors_layer_2], 1)
        # normalize the vector lengths
        descriptors = torch.nn.functional.normalize(descriptors,dim=1)
        # Repeat the descriptor vectors for the cut-off parts (for the lower dimensional layers)
        descriptors_ds_1 = torch.nn.functional.interpolate(descriptors.view(bs,self.c6*3,desc_h_0,desc_w_0),size=(desc_h_1,desc_w_1),mode='nearest-exact').view(bs, self.c6*3, -1)
        descriptors_ds_2 = torch.nn.functional.interpolate(descriptors.view(bs,self.c6*3,desc_h_0,desc_w_0),size=(desc_h_2,desc_w_2),mode='nearest-exact').view(bs, self.c6*3, -1)
        descriptors_repeated = torch.cat([descriptors,descriptors_ds_1,descriptors_ds_2], -1)
        return descriptors_repeated, descriptors

class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)
