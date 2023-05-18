import torch
import argparse
from model.IDO_model import IDOModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToPILImage
import numpy
from run_dataset import RunDataset
from utils.pytorch_msssim import ssim_matlab
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_bins', type=int, default=5, help='number of event stream bins')
parser.add_argument('--load_model', type=str, default='HsErgb.pth.pth', help='loade model path')  # ./et_model.pth
parser.add_argument('--n_skip', type=int, default=0, help='number of skip frames')
parser.add_argument('--num_inter', type=int, default=5, help='number of interpolated frames')
parser.add_argument('--root_path', type=str, default="./example", help='path of data')
parser.add_argument('--out_path', type=str, default='./out', help='the path of output video')
parser.add_argument('--n_control_point', type=int, default=5, help='the number of control point of motion spline')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def _interpolate(model, num_inter, skip, num_bins, root_path, out_path, mode="test"):
    inter_dataset = RunDataset(root=root_path, num_bins=num_bins, num_inter=num_inter, skip_frames=skip)
    inter_dataloader = torch.utils.data.DataLoader(
        inter_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)
    print("len_dataset: ", len(inter_dataset))
    count = 0
    with torch.no_grad():
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(tqdm(inter_dataloader)):
            input_images, voxel_grids_t_0, voxel_grids_t_1, voxel_grid_0_1, voxel_grid_1_0 = data
            input_images = [image.to(device) for image in input_images]  # [I0,I1]
            voxel_grids_t_0 = [event.to(device) for event in voxel_grids_t_0]  # [E_0_t1,E_0_t2,...]
            voxel_grids_t_1 = [event.to(device) for event in voxel_grids_t_1]  # [E_1_t1,E_1_t2,...]
            voxel_grid_0_1 = voxel_grid_0_1.to(device)  # E_0_1
            voxel_grid_1_0 = voxel_grid_1_0.to(device)  # E_1_0

            times_list = []
            for index in range(num_inter):
                times_list.append(float((index + 1) / (num_inter + 1)))

            outs, gate_mask, warp_I0_ts, warp_I1_ts, refine_I0_ts, refine_I1_ts = model(input_images, voxel_grids_t_0,
                                                                                        voxel_grids_t_1, voxel_grid_0_1,
                                                                                        voxel_grid_1_0,
                                                                                        times=times_list,
                                                                                        mode=mode)
            start_image = input_images[0][0, ...].cpu()
            start_image = ToPILImage()(start_image).convert("RGB")
            start_image.save(f"{out_path}/%06d.jpg" % count)
            count = count + 1
            for index in range(num_inter):
                out_img = torch.clamp(outs[index][0, ...].cpu(), 0, 1, )
                out_img = ToPILImage()(out_img).convert("RGB")
                out_img.save(f"{out_path}/%06d.jpg" % count)
                count = count + 1

        print("original video frames:", len(inter_dataset))
        print("interpolated video frames:", count - 1)
        print("argument:", "num_inter:", num_inter, "num_skip:", skip)


if __name__ == '__main__':

    opt = parser.parse_args()
    model = IDOModel(control_point=opt.n_control_point).cuda()

    if opt.load_model != '':
        model_dict = torch.load(opt.load_model)
        model.load_state_dict(model_dict)
        print('load_model:', opt.load_model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    _interpolate(model, skip=opt.n_skip, num_inter=opt.num_inter, num_bins=opt.num_bins,
                 root_path=opt.root_path, out_path=opt.out_path, mode="test")
