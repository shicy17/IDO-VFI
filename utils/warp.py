import torch as th
import numpy
import utils.pytorch_tools as pytorch_tools
#import pytorch_tools as pytorch_tools
def compute_source_coordinates(y_displacement, x_displacement):
    """Retruns source coordinates, given displacements.

    Given traget coordinates (y, x), the source coordinates are
    computed as (y + y_displacement, x + x_displacement).
    
    Args:
        x_displacement, y_displacement: are tensors with indices 
                                        [example_index, 1, y, x]
    """
    width, height = y_displacement.size(-1), y_displacement.size(-2)
    # torch.Size([856, 957])
    x_target, y_target = pytorch_tools.create_meshgrid(width, height,y_displacement.is_cuda)
    x_source = x_target + x_displacement.squeeze(1)  # 删掉为1的维度
    y_source = y_target + y_displacement.squeeze(1)

    out_of_boundary_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) |
                      (y_source.detach() < 0) | (y_source.detach() >= height))
    #print("x:",x_source,"y:",y_source)
    #print("mask",out_of_boundary_mask)
    # for i in range(out_of_boundary_mask.size()[0]):
    #     for j in range(out_of_boundary_mask.size()[1]):
    #         for k in range(out_of_boundary_mask.size()[2]):
    #             if out_of_boundary_mask[i][j][k]:
    #                 x_source[i][j][k] = x_target[j][k]
    #                 y_source[i][j][k] = y_target[j][k]
    # print(out_of_oundary_mask)
    return y_source, x_source, out_of_boundary_mask

def backwarp_2d(source, y_displacement, x_displacement):
    """Returns warped source image and occlusion_mask.
    Value in location (x, y) in output image in taken from
    (x + x_displacement, y + y_displacement) location of the source image.
    If the location in the source image is outside of its borders,
    the location in the target image is filled with zeros and the
    location is added to the "occlusion_mask".
    
    Args:
        source: is a tensor with indices
                [example_index, channel_index, y, x].
        x_displacement,
        y_displacement: are tensors with indices [example_index,
                        1, y, x]. 
    Returns:
        target: is a tensor with indices
                [example_index, channel_index, y, x].
        occlusion_mask: is a tensor with indices [example_index, 1, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    y_source, x_source, out_of_boundary_mask = compute_source_coordinates(
        y_displacement, x_displacement)
    x_source = (2.0 / float(width - 1)) * x_source - 1  # 归一化（-1,1）
    y_source = (2.0 / float(height - 1)) * y_source - 1
    # 偏移后的各像素点坐标，并且超出图外的坐标被置0
    x_source = x_source.masked_fill(out_of_boundary_mask, 0)
    y_source = y_source.masked_fill(out_of_boundary_mask, 0)   # 越界置为0
    #print("source", x_source, y_source)
    grid_source = th.stack([x_source, y_source], -1)
    # grid_source中的值对应的坐标，从source中取出
    #print(source)
    target = th.nn.functional.grid_sample(source,
                                          grid_source,
                                          align_corners=True)
    #print(target)
    out_of_boundary_mask = out_of_boundary_mask.unsqueeze(1)
    # True置为0
    target.masked_fill_(out_of_boundary_mask.expand_as(target), 0)

    return target, out_of_boundary_mask

def backwarp(source, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        source: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = source.size()
        # mesh grid
        xx = th.arange(0, W).view(1, -1).repeat(H, 1)
        yy = th.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = th.cat((xx, yy), 1).float()

        source = source.cuda()
        grid = grid.cuda()
        vgrid =th.autograd.Variable(grid) + flo  # B,2,H,W
        # 图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标

        # scale grid to [-1,1]
        ##2019 code
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0  # 取出光流u这个维度，同上

        vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
        output = th.nn.functional.grid_sample(source, vgrid, align_corners=True)
        mask = th.autograd.Variable(th.ones(source.size())).cuda()
        mask = th.nn.functional.grid_sample(mask, vgrid, align_corners=True)

        ##2019 author
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        ##2019 code
        # mask = torch.floor(torch.clamp(mask, 0 ,1))
        return output * mask

