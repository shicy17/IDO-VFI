import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Sep_STS_Encoder import ResBlock
from model.unet import UNet, UNet_60w
from utils.size_adapter import SizeAdapter as size_adapter
from model.Sep_STS_Encoder import SepSTSEncoder
import utils.warp as warp
import numpy
import math
from torchvision.transforms import ToPILImage
import time
from scipy import interpolate
import cv2
from utils.my_utils import to_subregion, gumbel_sigmoid, subregion_pad
from utils.softsplat import Softsplat


def _compute_weighted_average(attention, input_list):
    average = attention[:, 0, ...].unsqueeze(1) * input_list[0]
    for i in range(1, len(input_list)):
        average = average + attention[:, i, ...].unsqueeze(1) * input_list[i]
    return average


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


# Input: gate_mask:b,2,num_rows,num_cols x:b,c(22),H,W
class refine_warp_module(nn.Module):
    def __init__(self, div_rows, div_cols):
        super(refine_warp_module, self).__init__()
        self.div_rows = div_rows
        self.div_cols = div_cols
        self.num_regions = (2 * self.div_rows - 1) * (2 * self.div_cols - 1)
        self.flow_net = UNet(6 + 10 + 6, 4, False)
        self.attention_net = UNet_60w(4 * self.num_regions, self.num_regions, False)

    def forward(self, x, mode="train", gate_mask=None):
        num_rows = 2 * self.div_rows - 1
        num_cols = 2 * self.div_cols - 1
        b, c, h, w = x.size()
        device = x.device
        # get a list including local regions
        sub_resin_list = to_subregion(x, div_rows=self.div_rows, div_cols=self.div_cols,
                                      pad_to_original=False)  # b,c,H//2,W//2 len=num_region
        if gate_mask is None:
            gate_mask = torch.ones([b, 2, num_rows, num_cols]).to(device)
        if mode == "train":
            sub_resin = torch.cat(sub_resin_list, dim=0)  # b*num_region,c,H//2,W//2
            res_flows = self.flow_net(sub_resin)  # b*num_regions,c,h,w
            res_flows_list = []
            for i in range(self.num_regions):
                res_flows_list.append(res_flows[b * i:b * (i + 1), ...])
            index = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    mask = gate_mask[:, 1, i, j].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    res_flows_list[index] = mask * res_flows_list[index]
                    index = index + 1

            # local regions are padded to the size of full frame
            pad_res_list = subregion_pad(res_flows_list, [b, 4, h, w])
            pad_res_flows = torch.cat(pad_res_list, dim=1)
            # use an attention map to process overlap regions
            attention_scores = self.attention_net(pad_res_flows)
            attention = F.softmax(attention_scores, dim=1)  # b,num_region,h,w
            average = _compute_weighted_average(attention, pad_res_list)
            return average

        elif mode == "test":
            index = 0
            test_sub_resin_list = []
            for i in range(num_rows):
                for j in range(num_cols):
                    if gate_mask[:, 1, i, j] == 1:
                        test_sub_resin_list.append(sub_resin_list[index])
                    index = index + 1
            if len(test_sub_resin_list) > 0:
                sub_resin = torch.cat(test_sub_resin_list, dim=0)
                res_flows = self.flow_net(sub_resin)  # b*num_regions,c,h,w
                res_flows_list = []
                index = 0
                for i in range(num_rows):
                    for j in range(num_cols):
                        mask = gate_mask[:, 1, i, j].unsqueeze(1).unsqueeze(1).unsqueeze(1)
                        if gate_mask[:, 1, i, j] == 1:
                            # res flow of dynamic local region
                            res_flows_list.append(res_flows[b * index:b * (index + 1), ...] * mask)
                            index = index + 1
                        else:
                            # res flow of static local region is zero-vectors
                            res_flows_list.append(torch.zeros_like(res_flows[:b, ...] * mask).to(device))
                # local regions are padded to the size of full frame
                pad_res_list = subregion_pad(res_flows_list, [b, 4, h, w])
                pad_res_flows = torch.cat(pad_res_list, dim=1)
                # use an attention map to process overlap regions
                attention_scores = self.attention_net(pad_res_flows)
                attention = F.softmax(attention_scores, dim=1)
                average = _compute_weighted_average(attention, pad_res_list)
            else:
                average = torch.zeros([b, 4, h, w]).to(device)
            return average
        else:
            return None


class FlowGuidedGate(nn.Module):
    def __init__(self, in_planes, out_ch, out_size):
        super(FlowGuidedGate, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 128, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.AvgPool2d = nn.AdaptiveAvgPool2d((out_size[0], out_size[1]))
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=(3, 3), stride=(1, 1), bias=True, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.AvgPool2d(x)
        # x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn3(self.conv3(x))
        out = self.sigmoid(x)
        return out


def joinTensors(X1, X2, type="concat"):
    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.upconv = nn.ModuleList(
            [nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                                padding=1),
             ]
        )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x


class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                     ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class IDOModel(nn.Module):
    def __init__(self, joinType="concat", ks=5, dilation=1, control_point=2):
        super().__init__()

        nf = [192, 128, 64, 32]
        ws = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        nh = [2, 4, 8, 16]
        self.control_point = control_point
        self.joinType = joinType
        self.div_rows = 2
        self.div_cols = 2
        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)
        #
        self.frames_encoder = SepSTSEncoder(nf, NF=2, window_size=ws, nh=nh, in_ch=3 + 5)
        self.f_decoder = nn.Sequential(
            upSplit(nf[0], nf[1]),
            upSplit(nf[1] * growth, nf[2]),
            upSplit(nf[2] * growth, nf[3]),
        )

        def SmoothNet(inc, ouc):
            return torch.nn.Sequential(
                Conv_3d(inc, ouc, kernel_size=3, stride=1, padding=1, batchnorm=False),
                ResBlock(ouc, kernel_size=3),
            )

        nf_out = 64
        self.smooth = SmoothNet(nf[3] * growth, nf_out)
        self.predict = SynBlock(n_inputs=2, nf=nf_out, ks=ks, dilation=dilation, norm_weight=False)

        self.flow_model = UNet(6 + 10, 2 * self.control_point, False)
        self.fwarp = Softsplat()
        self.alpha = nn.Parameter(-torch.ones(1))
        self.gate1 = FlowGuidedGate(in_planes=4, out_ch=2, out_size=[2 * self.div_rows - 1, 2 * self.div_cols - 1])
        self.refine_module = refine_warp_module(div_rows=self.div_rows, div_cols=self.div_cols)

    def spline_inter(self, motion_spline, inter_t, mode='cubic'):
        b, c, H, W = motion_spline.size()
        device = motion_spline.device

        ts = numpy.linspace(0, 1, self.control_point + 1)

        motion_spline_x = motion_spline[:, :self.control_point, ...].cpu().detach().numpy()
        motion_spline_y = motion_spline[:, self.control_point:, ...].cpu().detach().numpy()
        cum_motion_spline_x = numpy.cumsum(motion_spline_x, 1)
        cum_motion_spline_y = numpy.cumsum(motion_spline_y, 1)
        cum_motion_spline_0 = numpy.zeros([b, 1, H, W])
        cum_motion_spline_x = numpy.concatenate((cum_motion_spline_0, cum_motion_spline_x), axis=1)
        cum_motion_spline_y = numpy.concatenate((cum_motion_spline_0, cum_motion_spline_y), axis=1)
        f_x = interpolate.interp1d(ts, cum_motion_spline_x, kind=mode, axis=1)
        f_y = interpolate.interp1d(ts, cum_motion_spline_y, kind=mode, axis=1)
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        x_t = torch.tensor(f_x(inter_t), dtype=torch.float).unsqueeze(1)
        y_t = torch.tensor(f_y(inter_t), dtype=torch.float).unsqueeze(1)
        flow_t = torch.cat([x_t, y_t], 1).to(device)
        return flow_t

    def forward(self, input_images, events_t_0, events_t_1, event_0_1, event_1_0, times, mode="train"):
        outs = []
        I0 = input_images[0]
        I1 = input_images[1]

        #   1.flow module
        motion_spine_0_1 = self.flow_model(torch.cat((I0, I1, event_0_1), dim=1))
        motion_spine_1_0 = self.flow_model(torch.cat((I1, I0, event_1_0), dim=1))

        cum_motion_spine_0_1_x = torch.cumsum(motion_spine_0_1[:, :self.control_point, :, :], 1)
        cum_motion_spine_0_1_y = torch.cumsum(motion_spine_1_0[:, self.control_point:, :, :], 1)
        flow_0_1 = torch.cat(
            [cum_motion_spine_0_1_x[:, -1, ...].unsqueeze(1), cum_motion_spine_0_1_y[:, -1, ...].unsqueeze(1)],
            1)  # b,2,h,w
        cum_motion_spine_1_0_x = torch.cumsum(motion_spine_1_0[:, :self.control_point, :, :], 1)
        cum_motion_spine_1_0_y = torch.cumsum(motion_spine_1_0[:, self.control_point:, :, :], 1)
        flow_1_0 = torch.cat(
            [cum_motion_spine_1_0_x[:, -1, ...].unsqueeze(1), cum_motion_spine_1_0_y[:, -1, ...].unsqueeze(1)], 1)

        for i in range(len(times)):
            t = times[i]
            event_t_0 = events_t_0[i]
            event_t_1 = events_t_1[i]
            # when training, just train the control point
            if mode == "train":
                time_point = math.ceil(t * self.control_point) - 1
                flow_0_t = torch.cat(
                    [cum_motion_spine_0_1_x[:, time_point, ...].unsqueeze(1),
                     cum_motion_spine_0_1_y[:, time_point, ...].unsqueeze(1)],
                    1)  # b,2,h,w
                time_point = math.ceil((1 - t) * self.control_point) - 1
                flow_1_t = torch.cat(
                    [cum_motion_spine_1_0_x[:, time_point, ...].unsqueeze(1),
                     cum_motion_spine_1_0_y[:, time_point, ...].unsqueeze(1)], 1)
            elif mode == "test":
                flow_0_t = self.spline_inter(motion_spine_0_1, t, mode='cubic')
                flow_1_t = self.spline_inter(motion_spine_1_0, 1 - t, mode='cubic')
            else:
                return None

            # softmax-splatting forward warping
            I1_0 = warp.backwarp(
                source=I1,
                flo=flow_0_1,
            )
            I0_1 = warp.backwarp(
                source=I0,
                flo=flow_1_0,
            )
            tenMetric_0 = torch.abs(I0 - I1_0)
            tenMetric_0 = self.alpha * torch.sum(tenMetric_0, dim=1, keepdim=True)
            tenMetric_1 = torch.abs(I1 - I0_1)
            tenMetric_1 = self.alpha * torch.sum(tenMetric_1, dim=1, keepdim=True)
            I0_t = self.fwarp(I0, flow_0_t, tenMetric_0)
            I1_t = self.fwarp(I1, flow_1_t, tenMetric_1)

            # 2.gate network
            sub_gatein = torch.cat([flow_0_t, flow_1_t], dim=1)
            gate_out = self.gate1(sub_gatein)  # b,2,num_rows,num_cols
            # gumbel-softmax trick is adopted when training
            if mode == "train":
                gate_mask = gumbel_sigmoid(gate_out, temperature=0.2, training=True)
            elif mode == "test":
                gate_mask = gumbel_sigmoid(gate_out, temperature=0.2, training=False)
                gate_mask[gate_mask >= 0.5] = 1
                gate_mask[gate_mask < 0.5] = 0  # b,2,num_rows,num_cols

            # 3.refine module
            res_input = torch.cat([I0, I1, I0_t, I1_t, event_t_0, event_t_1], dim=1)  # b*c*H*W
            res_flows = self.refine_module(res_input, gate_mask=gate_mask, mode=mode)  # b,4,H,
            res_flow_t_0 = res_flows[:, :2, :, :]
            res_flow_t_1 = res_flows[:, 2:, :, :]
            refine_I1_t = warp.backwarp(
                source=I1_t,
                flo=res_flow_t_1,
            )
            refine_I0_t = warp.backwarp(
                source=I0_t,
                flo=res_flow_t_0,
            )

            # 4.fusion module
            obj_images = torch.stack(
                [torch.cat([I0, event_t_0], dim=1), torch.cat([I1, event_t_1], dim=1)],
                dim=2)  # b*c*n_input*H*W
            mean_ = obj_images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
            obj_images = obj_images - mean_

            x_0, x_1, x_2, x_3, x_4 = self.frames_encoder(obj_images)
            dx_3 = self.lrelu(self.f_decoder[0](x_4, x_3.size()))
            dx_3 = joinTensors(dx_3, x_3, type=self.joinType)
            dx_2 = self.lrelu(self.f_decoder[1](dx_3, x_2.size()))
            dx_2 = joinTensors(dx_2, x_2, type=self.joinType)
            dx_1 = self.lrelu(self.f_decoder[2](dx_2, x_1.size()))
            dx_1 = joinTensors(dx_1, x_1, type=self.joinType)
            fea1 = self.smooth(dx_1)

            obj_frames = [refine_I0_t, refine_I1_t]
            out = self.predict(fea1, obj_frames, x_0.size()[-2:])
            outs.append(out)

        return outs, gate_mask, I0_t, I1_t, refine_I0_t, refine_I1_t


class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class SynBlock(nn.Module):
    def __init__(self, n_inputs, nf, ks, dilation, norm_weight=True):
        super(SynBlock, self).__init__()

        def Subnet_offset(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if norm_weight else nn.Identity()
            )

        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=nf, out_channels=n_inputs, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.n_inputs = n_inputs
        self.kernel_size = ks
        self.kernel_pad = int(((ks - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        import model.cupy_module.adacof as adacof
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.ModuleWeight = Subnet_weight(ks ** 2)
        self.ModuleAlpha = Subnet_offset(ks ** 2)
        self.ModuleBeta = Subnet_offset(ks ** 2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, output_size):
        H, W = output_size

        occ = torch.cat(torch.unbind(fea, 1), 1)
        occ = self.lrelu(self.feature_fuse(occ))
        Occlusion = self.moduleOcclusion(occ, (H, W))

        B, C, T, cur_H, cur_W = fea.shape
        fea = fea.transpose(1, 2).reshape(B * T, C, cur_H, cur_W)
        weights = self.ModuleWeight(fea, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(fea, (H, W)).view(B, T, -1, H, W)
        betas = self.ModuleBeta(fea, (H, W)).view(B, T, -1, H, W)

        warp = []
        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()
            alpha = alphas[:, i].contiguous()
            beta = betas[:, i].contiguous()
            occ = Occlusion[:, i:i + 1]
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode='bilinear')

            warp.append(
                occ * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation)
            )

        framet = sum(warp)
        return framet
