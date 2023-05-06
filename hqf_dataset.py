import h5py
import glob
import torch
import torchvision.transforms as transforms
import os
import numpy
from utils.event_utils import events_to_voxel_torch


def get_voxel_grid(xs, ys, ts, ps, num_bins, H, W):
    voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(H, W))
    return voxel_grid

class HqfDataset():
    def __init__(self, root, num_bins=5, num_inter=1):
        self.num_bins = num_bins
        self.root = root
        self.num_inter = num_inter
        self.imgs = []
        self.events = []
        self.start_image_index = []
        self.splite_dataset()

    def __getitem__(self, index):
        events_t_0 = []
        events_t_1 = []
        events_0_1 = []
        events_1_0 = []
        true_images = []

        index = self.start_image_index[index]
        start_image = self.imgs[index]
        end_image = self.imgs[index + self.num_inter + 1]
        transform_start_image = transforms.ToTensor()(start_image)
        transform_end_image = transforms.ToTensor()(end_image)
        input_images = [transform_start_image, transform_end_image]
        (H, W) = (transform_start_image.shape[-2], transform_start_image.shape[-1])
        for inter_index in range(self.num_inter):
            true_image = self.imgs[inter_index + index + 1]
            transform_true_image = transforms.ToTensor()(true_image)
            true_images.append(transform_true_image)

            event_t_0, event_t_1, event_0_1, event_1_0 = self.load_events_data(index,inter_index, H, W)
            events_t_0.append(event_t_0)
            events_t_1.append(event_t_1)
            events_0_1.append(event_0_1)
            events_1_0.append(event_1_0)
        return input_images, true_images, events_t_0, events_t_1, events_0_1[0],events_1_0[0]

    def load_events_data(self,index,inter_index, H, W):
        x, y, t, p = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        for before_event_index in range(index + 1, index + inter_index + 2):
            event_data = self.events[before_event_index]
            if len(event_data["t"]) > 0 and event_data["t"][-1] != event_data["t"][0]:
                x = numpy.hstack((x, event_data["x"].astype(numpy.float32).reshape((-1,))))
                y = numpy.hstack((y, event_data["y"].astype(numpy.float32).reshape((-1,))))
                t = numpy.hstack((t, event_data["t"].astype(numpy.float32).reshape((-1,))))
                p = numpy.hstack((p, event_data["p"].astype(numpy.float32).reshape((-1,))*2-1))

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
        for after_event_index in range(index + inter_index + 2, index + self.num_inter + 2):
            event_data = self.events[after_event_index]
            if len(event_data["t"]) > 0 and event_data["t"][-1] != event_data["t"][0]:
                x2 = numpy.hstack((x2, event_data["x"].astype(numpy.float32).reshape((-1,))))
                y2 = numpy.hstack((y2, event_data["y"].astype(numpy.float32).reshape((-1,))))
                t2 = numpy.hstack((t2, event_data["t"].astype(numpy.float32).reshape((-1,))))
                p2 = numpy.hstack((p2, event_data["p"].astype(numpy.float32).reshape((-1,))*2-1))
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
                                       torch.from_numpy(p_.copy().astype(numpy.float32)), 2 * self.num_bins, H, W)

            event_1_0 = get_voxel_grid(torch.from_numpy(reverse_x_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_y_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_t_.copy().astype(numpy.float32)),
                                       torch.from_numpy(reverse_p_.copy().astype(numpy.float32)), 2 * self.num_bins, H,
                                       W)
        else:
            event_0_1 = torch.zeros(2 * self.num_bins, H, W)
            event_1_0 = torch.zeros(2 * self.num_bins, H, W)
        return event_t_0, event_t_1, event_0_1, event_1_0

    def __len__(self):
        return len(self.start_image_index)



    def splite_dataset(self):
        h5_paths = sorted(glob.glob(os.path.join(self.root, "*.h5")))
        print(h5_paths)
        total_frame_index = 0
        for h5_path in h5_paths:
            f = h5py.File(h5_path, 'r')
            last_event_index = 0
            h5_index = 0
            ts=f["events/ts"][:]-f["events/ts"][0]
            for img_dataset in f["images"]:
                img = f["images"][img_dataset][:]
                # print(f["images"][img_dataset].attrs["timestamp"])
                self.imgs.append(img)
                if h5_index % (self.num_inter + 1) == 0 and (h5_index + self.num_inter + 1) < len(f["images"]):
                    self.start_image_index.append(total_frame_index)
                total_frame_index += 1
                event_index = f["images"][img_dataset].attrs["event_idx"]
                event_data = {'x': f["events/xs"][last_event_index:event_index + 1],
                              'y': f["events/ys"][last_event_index:event_index + 1],
                              't': ts[last_event_index:event_index + 1],
                              'p': f["events/ps"][last_event_index:event_index + 1]}
                self.events.append(event_data)
                last_event_index = event_index
                h5_index += 1


if __name__ == '__main__':
    validation_dataset = HqfDataset(root="D:\\LHX\\CVPR\\dataset\\Hqfdataset", num_inter=3)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)
    print(len(validation_dataset))
    for i, data in enumerate(validation_dataloader):
        print("  ")
