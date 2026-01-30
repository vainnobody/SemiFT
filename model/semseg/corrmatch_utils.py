import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange


class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(
            256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.conv2 = nn.Conv2d(
            256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, feature_in, out):
        dict_return = {}
        # Use feature_in shape for h_in/w_in logic from original code.
        # Original: h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        # Since divisor is 1, it's just shape.
        h_in, w_in = feature_in.shape[2], feature_in.shape[3]
        h_out, w_out = out.shape[2], out.shape[3]

        out = F.interpolate(
            out.detach(), (h_in, w_in), mode="bilinear", align_corners=True
        )
        feature = F.interpolate(
            feature_in, (h_in, w_in), mode="bilinear", align_corners=True
        )

        f1 = rearrange(self.conv1(feature), "n c h w -> n c (h w)")
        f2 = rearrange(self.conv2(feature), "n c h w -> n c (h w)")
        out_temp = rearrange(out, "n c h w -> n c (h w)")

        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(
            torch.tensor(f1.shape[1]).float()
        )
        corr_map = F.softmax(corr_map, dim=-1)

        corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)
        dict_return["corr_map"] = self.normalize_corr_map(
            corr_map_sample, h_in, w_in, h_out, w_out
        )
        dict_return["out"] = rearrange(
            torch.matmul(out_temp, corr_map), "n c (h w) -> n c h w", h=h_in, w=w_in
        )

        return dict_return

    def sample(self, corr_map, h_in, w_in):
        # Original code used 128 samples
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = rearrange(corr_map, "n m (h w) -> (n m) 1 h w", h=h_in, w=w_in)
        corr_map = F.interpolate(
            corr_map, (h_out, w_out), mode="bilinear", align_corners=True
        )

        corr_map = rearrange(corr_map, "(n m) 1 h w -> (n m) (h w)", n=n, m=m)
        range_ = (
            torch.max(corr_map, dim=1, keepdim=True)[0]
            - torch.min(corr_map, dim=1, keepdim=True)[0]
        )
        # Avoid division by zero if range_ is 0 (can happen with constant maps)
        range_ = range_ + 1e-6

        temp_map = ((-torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = temp_map > 0.5
        norm_corr_map = rearrange(
            corr_map, "(n m) (h w) -> n m h w", n=n, m=m, h=h_out, w=w_out
        )
        return norm_corr_map


class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
        if dist.is_available() and dist.is_initialized():
            self.gpu_num = dist.get_world_size()
        else:
            self.gpu_num = 1

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape

        if self.gpu_num > 1:
            pred_gather = torch.zeros([n * self.gpu_num, c, h, w]).cuda()
            dist.all_gather_into_tensor(pred_gather, pred)
            pred = pred_gather

            if ignore_mask is not None:
                ignore_mask_gather = torch.zeros([n * self.gpu_num, h, w]).cuda().long()
                dist.all_gather_into_tensor(ignore_mask_gather, ignore_mask)
                ignore_mask = ignore_mask_gather

        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        unique_cls = torch.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0

        for cls in unique_cls:
            cls_map = mask_pred == cls
            if ignore_mask is not None:
                cls_map *= ignore_mask != 255
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf

        if cls_num > 0:
            return_dict["new_global"] = new_global / cls_num
        else:
            return_dict["new_global"] = None

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g and thresh["new_global"] is not None:
            self.thresh_global = (
                self.momentum * self.thresh_global
                + (1 - self.momentum) * thresh["new_global"]
            )

    def get_thresh_global(self):
        return self.thresh_global
