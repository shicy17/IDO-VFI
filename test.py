import torch
import argparse
from model.IDO_model import IDOModel
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToPILImage
import numpy
from vimeo_dataset import VimeoDataset
from Hs_ergb_dataset import HsErgbDataset
from mb_dataset_allframes import MbDataset
from hqf_dataset import HqfDataset
from utils.pytorch_msssim import ssim_matlab
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_bins', type=int, default=5, help='number of event stream bins')
parser.add_argument('--load_model', type=str, default='hqf.pth', help='loade model path')  # ./et_model.pth
parser.add_argument('--n_skip', type=int, default=1, help='number of interpolated frames and skip frames')
parser.add_argument('--eval_dataset', type=str, default='Hqf', help='type of evaluation benchmark')
parser.add_argument('--n_control_point', type=int, default=5, help='the number of control point of motion spline')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def model_eval(model, skip, num_bins, eval_set, mode="test"):
    if eval_set == "Hqf":
        eval_dataset = HqfDataset(root="./dataset/Hqfdataset", num_bins=num_bins, num_inter=skip)

    elif eval_set == "middlebury":
        eval_dataset = MbDataset(root="D:\\LHX\\CVPR\\dataset\\Md_eval_allframes",
                                 num_bins=num_bins,
                                 skip_frames=skip,
                                 num_inter=skip)
    elif eval_set == "Hs_ergb_far":
        eval_dataset = HsErgbDataset(root="./dataset/hs_ergb_dataset",
                                     num_inter=skip, mode="far")
    elif eval_set == "Hs_ergb_close":
        eval_dataset = HsErgbDataset(root="./dataset/hs_ergb_dataset",
                                     num_inter=skip, mode="close")
    elif eval_set == "triplet_test" or eval_set == "septuplet_test":
        eval_dataset = VimeoDataset(tri_root="./dataset/vimeo_triplet",
                                    sep_root="./dataset/vimeo_septuplet",
                                    num_bins=num_bins,
                                    mode=eval_set,
                                    n_inputs=2,
                                    skip=skip)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)
    print("eval_dataset: ", eval_set, len(eval_dataset))
    with torch.no_grad():
        model.eval()
        ssim_score, PSNR = 0, 0
        ssim_list = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(tqdm(eval_dataloader)):
            input_images, true_images, events_t_0, events_t_1, event_0_1, event_1_0 = data
            input_images = [image.to(device) for image in input_images]  # [I0,I1]
            true_images = [image.to(device) for image in true_images]  # [It1,It2...]
            events_t_0 = [event.to(device) for event in events_t_0]  # [E_0_t1,E_0_t2,...]
            events_t_1 = [event.to(device) for event in events_t_1]  # [E_1_t1,E_1_t2,...]
            event_0_1 = event_0_1.to(device)  # E_0_1
            event_1_0 = event_1_0.to(device)  # E_1_0

            times_list = []
            for index in range(skip):
                times_list.append(float((index + 1) / (len(true_images) + 1)))

            outs, gate_mask, warp_I0_t, warp_I1_t, refine_I0_t, refine_I1_t = model(input_images, events_t_0,
                                                                                    events_t_1, event_0_1, event_1_0,
                                                                                    times=times_list,
                                                                                    mode=mode)
            for index in range(skip):
                out_img = torch.clamp(outs[index][0, ...].cpu(), 0, 1, )
                out_img = ToPILImage()(out_img).convert("RGB")

                label = true_images[index][0, ...].cpu()
                label = ToPILImage()(label).convert("RGB")

                ssim_score = ssim_score + compare_ssim(numpy.array(out_img),
                                                       numpy.array(label),
                                                       data_range=255, multichannel=True)
                PSNR = PSNR + compare_psnr(numpy.array(out_img),
                                           numpy.array(label),
                                           data_range=255)

                ssim = ssim_matlab(torch.round(true_images[index][0, ...] * 255).unsqueeze(0) / 255.,
                                   torch.round(outs[index][0, ...] * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

                ssim_list.append(ssim)

        print("matlab_ssim:", numpy.mean(ssim_list))
        print("skimage.metrics_ssim:", ssim_score / skip / len(eval_dataset), "skimage.metrics_psnr:",
              PSNR / skip / len(eval_dataset))


if __name__ == '__main__':

    opt = parser.parse_args()
    model = IDOModel(control_point=opt.n_control_point).cuda()

    if opt.load_model != '':
        model_dict = torch.load(opt.load_model)
        model.load_state_dict(model_dict)
        print('load_model:', opt.load_model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))
    model_eval(model, skip=opt.n_skip, num_bins=opt.num_bins, eval_set=opt.eval_dataset, mode="test")
