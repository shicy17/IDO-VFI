import torch
import argparse
from model.IDO_model import IDOModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToPILImage
import numpy
from vimeo90k_dataset import Vimeo90kDataset
from Hs_ergb_dataset import HsErgbDataset
from hqf_dataset import HqfDataset
from utils.pytorch_msssim import ssim_matlab
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_bins', type=int, default=5, help='number of event stream bins')
parser.add_argument('--load_model', type=str, default='hqf.pth', help='loade model path')  # ./et_model.pth
parser.add_argument('--n_skip', type=int, default=5, help='number of interpolated frames and skip frames')
parser.add_argument('--eval_dataset', type=str, default='Hs_ergb_far', help='type of evaluation benchmark')
parser.add_argument('--n_control_point', type=int, default=5, help='the number of control point of motion spline')
parser.add_argument('--data_path', type=str, default="./dataset/Hqfdataset", help='path of data')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def model_eval(model, skip, num_bins, eval_set, data_path, mode="test"):
    if eval_set == "Hqf":
        eval_dataset = HqfDataset(root=data_path, num_bins=num_bins,
                                  skip_frames=skip, num_inter=skip, label_require=True)
    elif eval_set == "Hs_ergb_far":
        eval_dataset = HsErgbDataset(root=data_path, num_bins=num_bins,
                                     skip_frames=skip, num_inter=skip, mode="far", label_require=True)
    elif eval_set == "Hs_ergb_close":
        eval_dataset = HsErgbDataset(root=data_path, num_bins=num_bins,
                                     skip_frames=skip, num_inter=skip, mode="close", label_require=True)
    elif eval_set == "triplet_test" or eval_set == "septuplet_test":
        eval_dataset = Vimeo90kDataset(root=data_path,
                                       num_bins=num_bins,
                                       mode=eval_set,
                                       num_inter=skip,
                                       skip=skip,
                                       label_require=True)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4)
    print("eval_dataset: ", eval_set, len(eval_dataset))
    with torch.no_grad():
        model.eval()
        ssim_score, warp_ssim_score, refine_ssim_score, PSNR, warp_PSNR, refine_PSNR = 0, 0, 0, 0, 0, 0
        ssim_list = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(tqdm(eval_dataloader)):
            input_images, true_images, voxel_grids_t_0, voxel_grids_t_1, voxel_grid_0_1, voxel_grid_1_0 = data

            input_images = [image.to(device) for image in input_images]  # [I0,I1]
            true_images = [image.to(device) for image in true_images]  # [It1,It2...]
            voxel_grids_t_0 = [event.to(device) for event in voxel_grids_t_0]  # [E_0_t1,E_0_t2,...]
            voxel_grids_t_1 = [event.to(device) for event in voxel_grids_t_1]  # [E_1_t1,E_1_t2,...]
            voxel_grid_0_1 = voxel_grid_0_1.to(device)  # E_0_1
            voxel_grid_1_0 = voxel_grid_1_0.to(device)  # E_1_0

            times_list = []
            for index in range(skip):
                times_list.append(float((index + 1) / (len(true_images) + 1)))

            outs, gate_mask, warp_I0_ts, warp_I1_ts, refine_I0_ts, refine_I1_ts = model(input_images, voxel_grids_t_0,
                                                                                        voxel_grids_t_1, voxel_grid_0_1,
                                                                                        voxel_grid_1_0,
                                                                                        times=times_list,
                                                                                        mode=mode)
            for index in range(skip):
                label = true_images[index][0, ...].cpu()
                label = ToPILImage()(label).convert("RGB")
                out_img = torch.clamp(outs[index][0, ...].cpu(), 0, 1, )
                out_img = ToPILImage()(out_img).convert("RGB")
                warp_I0_t_img = torch.clamp(warp_I0_ts[index][0, ...].cpu(), 0, 1, )
                warp_I0_t_img = ToPILImage()(warp_I0_t_img).convert("RGB")
                warp_I1_t_img = torch.clamp(warp_I1_ts[index][0, ...].cpu(), 0, 1, )
                warp_I1_t_img = ToPILImage()(warp_I1_t_img).convert("RGB")
                refine_I0_t_img = torch.clamp(refine_I0_ts[index][0, ...].cpu(), 0, 1, )
                refine_I0_t_img = ToPILImage()(refine_I0_t_img).convert("RGB")
                refine_I1_t_img = torch.clamp(refine_I1_ts[index][0, ...].cpu(), 0, 1, )
                refine_I1_t_img = ToPILImage()(refine_I1_t_img).convert("RGB")
                ssim_score = ssim_score + compare_ssim(numpy.array(out_img),
                                                       numpy.array(label),
                                                       data_range=255, channel_axis=-1)
                PSNR = PSNR + compare_psnr(numpy.array(out_img),
                                           numpy.array(label),
                                           data_range=255)

                warp_ssim_score = warp_ssim_score + compare_ssim(numpy.array(warp_I0_t_img),
                                                                 numpy.array(label),
                                                                 data_range=255,  channel_axis=-1) + \
                                  compare_ssim(numpy.array(warp_I1_t_img), numpy.array(label), data_range=255,
                                               channel_axis=-1)
                warp_PSNR = warp_PSNR + compare_psnr(numpy.array(warp_I0_t_img),
                                                     numpy.array(label),
                                                     data_range=255) + \
                            compare_psnr(numpy.array(warp_I1_t_img),
                                         numpy.array(label),
                                         data_range=255)

                refine_ssim_score = refine_ssim_score + compare_ssim(numpy.array(refine_I0_t_img),
                                                                     numpy.array(label),
                                                                     data_range=255,  channel_axis=-1) + \
                                    compare_ssim(numpy.array(refine_I1_t_img), numpy.array(label), data_range=255,
                                                 channel_axis=-1)
                refine_PSNR = refine_PSNR + compare_psnr(numpy.array(refine_I0_t_img),
                                                         numpy.array(label),
                                                         data_range=255) + \
                              compare_psnr(numpy.array(refine_I1_t_img),
                                           numpy.array(label),
                                           data_range=255)

                ssim = ssim_matlab(torch.round(true_images[index][0, ...] * 255).unsqueeze(0) / 255.,
                                   torch.round(outs[index][0, ...] * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

                ssim_list.append(ssim)

        print("matlab_ssim:")
        print("final out", numpy.mean(ssim_list))
        print("=" * 30)
        print("skimage.metrics:")
        print("final out ssim:", ssim_score / skip / len(eval_dataset), "final out psnr:",
              PSNR / skip / len(eval_dataset))
        print("flow module out ssim:", warp_ssim_score / 2 / skip / len(eval_dataset), "flow module out psnr:",
              warp_PSNR / 2 / skip / len(eval_dataset))
        print("refine module out ssim:", refine_ssim_score / 2 / skip / len(eval_dataset), "refine module psnr:",
              refine_PSNR / 2 / skip / len(eval_dataset))


if __name__ == '__main__':

    opt = parser.parse_args()
    model = IDOModel(control_point=opt.n_control_point).cuda()

    if opt.load_model != '':
        model_dict = torch.load(opt.load_model)
        model.load_state_dict(model_dict)
        print('load_model:', opt.load_model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    model_eval(model, data_path=opt.data_path, skip=opt.n_skip, num_bins=opt.num_bins, eval_set=opt.eval_dataset,
               mode="test")
