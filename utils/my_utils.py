import torch
import torch.nn.functional as F
from torchvision import models
from thop import profile
import torch.nn as nn
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb

def flow_to_image(flow, max_flow=256):
    flow=flow.cpu().permute(1, 2, 0).numpy()

    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)
    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)

def gumbel_sigmoid(x, temperature, training):
    """Apply gumbel-sigmoid to input x.

    Args:
      x: Tensor, input tensor.
      temperature: float, temperature for gumbel-softmax.
      training: boolean, whether in training mode.

    Returns:
      Tensor, output tensor with gumbel-sigmoid applied.
    """
    # Generate Gumbel noise
    shape = x.shape
    uniform_noise = torch.rand(shape).to(x.device)
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-20) + 1e-20)

    if training:
        # Apply Gumbel-Softmax trick
        logits = (x + gumbel_noise) / temperature
        y = F.softmax(logits, dim=1)
    else:
        # Use Sigmoid function at inference time
        logits = x / temperature
        y = F.softmax(logits, dim=1)
        # y = torch.sigmoid(logits)

    return y


''' the height and width of input should be divided by div_rows and div_cols separately
    the window size equals height and width of input divided by div_rows and  div_cols
    the input size ...,H,W
    return Tensor list,the length of list equals (div_rows*2-1)*(div_cols*2-1) 
        the shape of sub region tensor tensor of the list [...,H//div_rows,W//div_cols]
        the shape of padded sub region tensor of the list [...,H,W]
'''


def to_subregion(inputs, div_rows=2, div_cols=2, stride=None,pad_to_original=False,):
    h, w = inputs.size()[-2:]
    window_size = [h // div_rows, w // div_cols]
    if stride==None:
        stride = [window_size[0] // 2, window_size[1] // 2]
        num_rows = div_rows * 2 - 1
        num_cols = div_cols * 2 - 1
    else:
        num_rows = div_rows
        num_cols = div_cols
    sub_regions = []
    padded_sub_regions = []
    for i in range(num_rows):
        for j in range(num_cols):
            sub_region = inputs[..., stride[0] * i:stride[0] * i + window_size[0],
                         stride[1] * j:stride[1] * j + window_size[1]]
            # print(stride[0] * i,stride[0] * i + window_size[0]," ",stride[1] * j,stride[1] * j + window_size[1])
            sub_regions.append(sub_region)
            if pad_to_original:
                padded_sub_region = torch.zeros_like(inputs.size()).to(inputs.device)
                padded_sub_region[..., stride[0] * i:stride[0] * i + window_size[0],
                stride[1] * j:stride[1] * j + window_size[1]] = sub_region
                padded_sub_regions.append(padded_sub_region)
    if pad_to_original:
        return sub_regions, padded_sub_regions
    else:
        return sub_regions


''' 
    it is reverse operation against to_subregion,the tensor of input list will be arranged into it's original location,
    and padded into pad_size (zero padding)
    return Tensor list,the length of list equals length of input_list 
        the shape of padded sub region tensor of the list [...,H,W]
'''


def subregion_pad(input_list, pad_size,div_rows=2, div_cols=2, stride=None):
    h, w = pad_size[-2:]
    window_size = [h // div_rows, w // div_cols]
    if stride==None:
        stride = [window_size[0] // 2, window_size[1] // 2]
        num_rows = div_rows * 2 - 1
        num_cols = div_cols * 2 - 1
    else:
        num_rows = div_rows
        num_cols = div_cols
    padded_sub_regions = []
    index = 0
    device = input_list[0].device
    blank_zero = torch.zeros(pad_size)
    # print(input_list[0].size(),pad_size)

    for i in range(num_rows):
        for j in range(num_cols):
            # padded_sub_region=torch.zeros(pad_size).to(device)
            padded_sub_region=blank_zero.to(device)
            # print(padded_sub_region.size(),input_list[index].size(),stride,window_size)
            padded_sub_region[..., stride[0] * i:stride[0] * i + window_size[0],
            stride[1] * j:stride[1] * j + window_size[1]] = input_list[index]
            padded_sub_regions.append(padded_sub_region)
            index = index + 1
    return padded_sub_regions


def cal_subregion_flops(net, size, div_rows, div_cols):
    h, w = size[-2:]
    sub_region_h = h // div_rows
    sub_region_w = w // div_cols
    inputs = torch.randn([1, 22, sub_region_h, sub_region_w])
    flops, params = profile(net, inputs=(inputs,))
    flops_per_pixel = flops / (sub_region_h * sub_region_w)
    return flops_per_pixel


class GateLoss(nn.Module):
    def __init__(self, factor):
        super(GateLoss, self).__init__()
        self.factor = factor

    def forward(self, x):
        # x:b,2,h,w
        y = torch.zeros_like(x)
        y[:, 0, ...] = 1
        loss = abs(x - y)
        loss = loss.sum()
        return loss * self.factor


if __name__ == '__main__':
    input = torch.randn(5, 4).cuda()
    print(input)
    sub_regions = to_subregion(input, div_rows=2, div_cols=1)
    print(sub_regions)
    padded_sub_regions = subregion_pad(sub_regions, pad_size=input.size(), div_rows=2, div_cols=1)
    print(padded_sub_regions)
    print(sub_regions[0].size())
