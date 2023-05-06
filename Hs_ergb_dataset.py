import numpy
import os
import torch
import torch.utils.data
from tqdm import tqdm
from PIL import Image
from utils.event_utils import events_to_voxel_torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import sys
import glob


# return (num_bins,H,W)
def get_voxel_grid(xs, ys, ts, ps, num_bins, H, W):
    # hot_events_mask = numpy.ones([num_bins, H,W])
    # hot_events_mask = torch.from_numpy(hot_events_mask).float()
    # generate voxel grid which has size self.num_bins x H x W
    voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(H, W))
    # print('voxel',voxel_grid.size())
    # voxel_grid = voxel_grid * hot_events_mask
    return voxel_grid


class HsErgbDataset():
    def __init__(self, root, num_bins=5, skip_frames=1, num_inter=1, mode="close"):
        self.num_bins = num_bins
        self.root = root
        self.mode = mode
        self.skip_frames = skip_frames
        self.num_inter = num_inter
        self.data_dic = []
        self.split_dataset()

    def __getitem__(self, index):
        events_t_0 = []
        events_t_1 = []
        events_0_1 = []
        events_1_0 = []
        true_images = []

        data = self.data_dic[index]
        I0 = Image.open(data["start_image"])
        I1 = Image.open(data["end_image"])
        I0 = transforms.ToTensor()(I0)
        I1 = transforms.ToTensor()(I1)
        input_images = [I0, I1]
        (H, W) = (I0.shape[-2], I1.shape[-1])
        for inter_index in range(self.num_inter):
            inter_data = data["mid"][inter_index]
            true_image = transforms.ToTensor()(Image.open(inter_data["true_image"]))
            true_images.append(true_image)
            event_t_0, event_t_1, event_0_1, event_1_0 = self.load_events_data(inter_data["before_events"],
                                                                               inter_data["after_events"], H, W)
            events_t_0.append(event_t_0)
            events_t_1.append(event_t_1)
            events_0_1.append(event_0_1)
            events_1_0.append(event_1_0)
        return input_images, true_images, events_t_0, events_t_1, events_0_1[0],events_1_0[0]

    def load_events_data(self, before_event_files, after_event_files, H, W):
        x, y, t, p = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        for before_event_file in before_event_files:
            try:
                event_data = numpy.load(before_event_file)
            except:
                print("can't load file:", before_event_file)
                continue
            if len(event_data["t"]) > 0 and event_data["t"][-1] != event_data["t"][0]:
                x = numpy.hstack((x, event_data["x"].astype(numpy.float32).reshape((-1,))))
                y = numpy.hstack((y, event_data["y"].astype(numpy.float32).reshape((-1,))))
                t = numpy.hstack((t, event_data["t"].astype(numpy.float32).reshape((-1,))))
                p = numpy.hstack((p, event_data["p"].astype(numpy.float32).reshape((-1,)) * 2 - 1))

        if 0 < len(t) == len(y) and len(x) == len(y) and len(t) == len(p):
            mask = (x <= W - 1) & (y <= H - 1) & (x >= 0) & (y >= 0)
            x_ = x[mask]
            y_ = y[mask]
            p_ = p[mask]
            t_ = t[mask]
            t_ = t_.max() + t_.min() - t_
            t_ = t_[::-1]
            x_ = x_[::-1]
            y_ = y_[::-1]
            p_ = -p_[::-1]
            event_t_0 = get_voxel_grid(torch.from_numpy(x_.copy().astype(numpy.float32)),
                                       torch.from_numpy(y_.copy().astype(numpy.float32)),
                                       torch.from_numpy(t_.copy().astype(numpy.float32)),
                                       torch.from_numpy(p_.copy().astype(numpy.float32)), self.num_bins, H, W)
        else:
            event_t_0 = torch.zeros(self.num_bins, H, W)

        x2, y2, t2, p2 = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        for after_event_file in after_event_files:
            try:
                event_data = numpy.load(after_event_file)
            except:
                print("can't load file:", after_event_file)
                continue
            if len(event_data["t"]) > 0 and event_data["t"][-1] != event_data["t"][0]:
                x2 = numpy.hstack((x2, event_data["x"].astype(numpy.float32).reshape((-1,))))
                y2 = numpy.hstack((y2, event_data["y"].astype(numpy.float32).reshape((-1,))))
                t2 = numpy.hstack((t2, event_data["t"].astype(numpy.float32).reshape((-1,))))
                p2 = numpy.hstack((p2, event_data["p"].astype(numpy.float32).reshape((-1,)) * 2 - 1))
        if 0 < len(t2) == len(y2) and len(x2) == len(y2) and len(t2) == len(p2):
            mask = (x2 <= W - 1) & (y2 <= H - 1) & (x2 >= 0) & (y2 >= 0)
            x_ = x2[mask]
            y_ = y2[mask]
            p_ = p2[mask]
            t_ = t2[mask]
            event_t_1 = get_voxel_grid(torch.from_numpy(x_.copy().astype(numpy.float32)),
                                       torch.from_numpy(y_.copy().astype(numpy.float32)),
                                       torch.from_numpy(t_.copy().astype(numpy.float32)),
                                       torch.from_numpy(p_.copy().astype(numpy.float32)), self.num_bins, H, W)
        else:
            event_t_1 = torch.zeros(self.num_bins, H, W)

        total_x = numpy.hstack((x, x2))
        total_y = numpy.hstack((y, y2))
        total_t = numpy.hstack((t, t2))
        total_p = numpy.hstack((p, p2))
        if 0 < len(total_t) == len(total_x) and len(total_x) == len(total_y) and len(total_y) == len(total_p):
            mask = (total_x <= W - 1) & (total_y <= H - 1) & (total_x >= 0) & (total_y >= 0)
            x_ = total_x[mask]
            y_ = total_y[mask]
            p_ = total_p[mask]
            t_ = total_t[mask]
            reverse_t_ = t_.max() + t_.min() - t_
            reverse_t_ = reverse_t_[::-1]
            reverse_x_ = x_[::-1]
            reverse_y_ = y_[::-1]
            reverse_p_ = -p_[::-1]
            event_0_1 = get_voxel_grid(torch.from_numpy(x_.copy().astype(numpy.float32)),
                                       torch.from_numpy(y_.copy().astype(numpy.float32)),
                                       torch.from_numpy(t_.copy().astype(numpy.float32)),
                                       torch.from_numpy(p_.copy().astype(numpy.float32)), 2*self.num_bins, H, W)

            event_1_0 = get_voxel_grid(torch.from_numpy(reverse_x_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_y_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_t_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_p_.copy().astype(numpy.float32)), 2*self.num_bins, H, W)
        else:
            event_0_1 = torch.zeros(2 * self.num_bins, H, W)
            event_1_0 = torch.zeros(2 * self.num_bins, H, W)
        return event_t_0, event_t_1, event_0_1, event_1_0

    def __len__(self):
        return len(self.data_dic)

    def split_dataset(self):
        if self.mode == "close":
            self.root = os.path.join(self.root, "close/test")
        elif self.mode == "far":
            self.root = os.path.join(self.root, "far/test")
        dirs = os.listdir(self.root)
        print(self.root, dirs)
        for dir in dirs:
            event_root = os.path.join(self.root, dir, "events_aligned")
            img_root = os.path.join(self.root, dir, "images_corrected")
            time_stamp_path = os.path.join(self.root, 'images_corrected', 'timestamp.txt')
            image_files = sorted(glob.glob(os.path.join(img_root, "*.png")))
            event_files = sorted(glob.glob(os.path.join(event_root, "*.npz")))
            # print(os.path.join(event_root,os.path.relpath(path,img_root),"*.npz"))
            for i in range(int((len(event_files) - 1) / (self.num_inter + 1))):
                start_img_index = i * (self.num_inter + 1)
                data = {"start_image": image_files[start_img_index],
                        "end_image": image_files[start_img_index + self.num_inter + 1],
                        "mid": []}
                for j in range(self.num_inter):
                    inter_before_events = []
                    inter_after_events = []

                    for before_event_index in range(start_img_index + 1, start_img_index + j + 2):
                        inter_before_events.append(event_files[before_event_index])
                    for after_event_index in range(start_img_index + j + 2, start_img_index + self.num_inter + 2):
                        inter_after_events.append(event_files[after_event_index])

                    data["mid"].append(
                        {"true_image": image_files[start_img_index + j + 1], "before_events": inter_before_events,
                         "after_events": inter_after_events})
                self.data_dic.append(data)


if __name__ == '__main__':

    validation_dataset = HsErgbDataset(root="D:\\LHX\\事件驱动型Transformer\\Flow0\\2FlowMultiGpu\\hs_ergb_dataset",
                                       num_inter=3, mode="far")
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)
    print(len(validation_dataset))
    for i, data in enumerate(validation_dataloader):
        print("  ")
