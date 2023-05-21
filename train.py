import torch
import os
from model.IDO_model import IDOModel, refine_warp_module
import argparse
from vimeo90k_dataset import Vimeo90kDataset
import torch.optim as optim
import time
from test import model_eval
from Hs_ergb_dataset import HsErgbDataset
from hqf_dataset import HqfDataset
from utils.my_utils import cal_subregion_flops, GateLoss

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=4, help='input batch size')
parser.add_argument(
    '--num_bins', type=int, default=5, help='number of event stream bins')
parser.add_argument(
    '--nepoch', type=int, default=15, help='number of epochs to train for')
parser.add_argument('--dataset_path', default='./dataset/vimeo_triplet', type=str, help="dataset path")
parser.add_argument('--load_model', type=str, default='', help='loade model path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--n_skip', type=int, default=1, help='number of interpolated frames and skip frames')
parser.add_argument('--train_module', type=int, default=0, help='the submodule should be trained')  # train the model step by step
parser.add_argument('--n_control_point', type=int, default=5, help='the number of control point of optical flow')

if __name__ == '__main__':
    opt = parser.parse_args()

    model = IDOModel(control_point=opt.n_control_point).cuda()
    if opt.train_module == 0:  # flow module
        for param in model.parameters():
            param.requires_grad = False
        for param in model.flow_model.parameters():
            param.requires_grad = True
        for param in model.fwarp.parameters():
            param.requires_grad = True
        model.alpha.requires_grad = True
    elif opt.train_module == 1:  # refine module+gate
        for param in model.parameters():
            param.requires_grad = False
        for param in model.refine_module.parameters():
            param.requires_grad = True
        for param in model.gate1.parameters():
            param.requires_grad = True
    elif opt.train_module == 2:  # fusion module
        for param in model.parameters():
            param.requires_grad = True
        for param in model.flow_model.parameters():
            param.requires_grad = False
        for param in model.refine_module.parameters():
            param.requires_grad = False
        for param in model.gate1.parameters():
            param.requires_grad = False
        for param in model.fwarp.parameters():
            param.requires_grad = False
        model.alpha.requires_grad = False

    if torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if opt.load_model != '':
        model_dict = torch.load(opt.load_model)
        model.load_state_dict(model_dict)
        print('load_model:', opt.load_model)

    train_dataset = Vimeo90kDataset(root=opt.dataset_path,
                                    num_bins=opt.num_bins,
                                    mode='triplet_train',
                                    num_inter=opt.n_skip,
                                    skip=opt.n_skip,
                                    label_require=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=4)

    print("train_dataset:", len(train_dataset))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    l1_loss_fn = torch.nn.L1Loss(size_average=False)
    perceptual_loss = torch.nn.MSELoss(size_average=False)

    div_rows = 2
    div_cols = 2
    input_images, true_images, voxel_grids_t_0, voxel_grids_t_1, voxel_grid_0_1, voxel_grid_1_0 = train_dataset[0]
    refine_net = refine_warp_module(div_rows=div_rows, div_cols=div_cols)
    FLOPs_per_pixel = cal_subregion_flops(refine_net, input_images[0].size(), div_rows, div_cols)
    lamda =  5*1E-5
    gate_loss_fn = GateLoss(FLOPs_per_pixel/lamda)

    # optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(opt.nepoch):
        train_losses, out_losses, gate_losses = 0, [0, 0], 0
        model.train()
        start_time = time.time()
        for i, data in enumerate(train_dataloader):
            input_images, true_images, events_t_0, events_t_1, event_0_1, event_1_0 = data
            input_images = [image.to(device) for image in input_images]  # [I0,I1]
            true_images = [image.to(device) for image in true_images]  # [It1,It2...]
            events_t_0 = [event.to(device) for event in events_t_0]  # [E_0_t1,E_0_t2,...]
            events_t_1 = [event.to(device)for event in events_t_1]  # [E_1_t1,E_1_t2,...]
            event_0_1 = event_0_1.to(device) # E_0_1
            event_1_0 = event_1_0.to(device) # E_1_0

            for index in range(len(true_images)):
                true_image = true_images[index]
                outs, gate_mask, warp_I0_ts, warp_I1_ts, refine_I0_ts, refine_I1_ts = model(input_images, events_t_0,
                                                                                            events_t_1, event_0_1,
                                                                                            event_1_0,
                                                                                            times=[float((index + 1) / (
                                                                                                    len(
                                                                                                        true_images) + 1))],
                                                                                            mode="train")
                if opt.train_module == 0:
                    train_loss = (l1_loss_fn(true_image, warp_I0_ts[0]) + perceptual_loss(true_image,
                                                                                          warp_I0_ts[0]) + l1_loss_fn(
                        true_image, warp_I1_ts[0]) + perceptual_loss(true_image, warp_I1_ts[0])) / 2
                elif opt.train_module == 1:
                    reconstruct_loss = (l1_loss_fn(true_image, refine_I0_ts[0]) + perceptual_loss(true_image,
                                                                                                  refine_I0_ts[
                                                                                                      0]) + l1_loss_fn(
                        true_image, refine_I1_ts[0]) + perceptual_loss(true_image, refine_I1_ts[0])) / 2
                    gate_loss = gate_loss_fn(gate_mask)
                    train_loss = reconstruct_loss + gate_loss
                elif opt.train_module == 2:
                    train_loss = perceptual_loss(true_image, outs[0]) + l1_loss_fn(true_image, outs[0])

                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses = train_loss.item() + train_losses

        scheduler.step()
        print("=" * 30)
        print('epoch', epoch, "train_loss:", train_losses / (len(train_dataset)))
        print("gate_loss:", gate_losses / (len(train_dataset)))
        print('time per epoch:', time.time() - start_time)
        torch.save(model.state_dict(), './checkpoint.pth')
