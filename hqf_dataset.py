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
    def __init__(self, root, num_bins=5, skip_frames=1, num_inter=1, label_require=True):
        self.skip_frames = skip_frames
        self.label_require = label_require
        self.num_bins = num_bins
        self.root = root
        self.num_inter = num_inter
        self.data_dic = []
        self.splite_dataset()

    def __getitem__(self, index):
        data = self.data_dic[index]
        I0 = data["start_image"]
        I1 = data["end_image"]
        I0 = transforms.ToTensor()(I0)
        I1 = transforms.ToTensor()(I1)
        input_images = [I0, I1]
        start_time = data["timestamps"][0]
        end_time = data["timestamps"][1]
        (H, W) = (I0.shape[-2], I1.shape[-1])
        event_0_1 = data["event_sequences"]
        event_0_1[:,3]=event_0_1[:,3]*2 - 1 # polarity [0,1]>[-1,1]
        voxel_grid_0_1 = self.event_sequence_to_voxel_grid(event_0_1, start_time, end_time, 2 * self.num_bins, H, W,
                                                           reverse=False)
        voxel_grid_1_0 = self.event_sequence_to_voxel_grid(event_0_1, start_time, end_time, 2 * self.num_bins, H, W,
                                                           reverse=True)
        voxel_grids_t_0 = []
        voxel_grids_t_1 = []
        true_images = []
        mid_timestamps = numpy.linspace(start_time, end_time, self.num_inter + 2)[1:-1]
        for inter_index in range(self.num_inter):
            if self.label_require:
                inter_data = data["mid"][inter_index]
                mid_timestamp = inter_data["mid_timestamp"]
                true_image = transforms.ToTensor()(inter_data["true_image"])
                true_images.append(true_image)
            else:
                mid_timestamp = mid_timestamps[inter_index]
            event_0_t_mask = (start_time <= event_0_1[:, 2]) & (mid_timestamp > event_0_1[:, 2])

            event_0_t = event_0_1[event_0_t_mask]
            event_t_1_mask = (mid_timestamp <= event_0_1[:, 2]) & (end_time > event_0_1[:, 2])
            event_t_1 = event_0_1[event_t_1_mask]
            voxel_grid_t_0 = self.event_sequence_to_voxel_grid(event_0_t, start_time, mid_timestamp, self.num_bins, H,
                                                               W,
                                                               reverse=True)
            voxel_grid_t_1 = self.event_sequence_to_voxel_grid(event_t_1, mid_timestamp, end_time, self.num_bins, H, W,
                                                               reverse=False)
            voxel_grids_t_0.append(voxel_grid_t_0)
            voxel_grids_t_1.append(voxel_grid_t_1)
        return input_images, true_images, voxel_grids_t_0, voxel_grids_t_1, voxel_grid_0_1, voxel_grid_1_0

    def event_sequence_to_voxel_grid(self, event_sequence, start_time, end_time, num_bins, H, W, reverse=False):
        x = event_sequence[:, 0]
        y = event_sequence[:, 1]
        t = event_sequence[:, 2]
        p = event_sequence[:, 3]
        try:
            dt = t[-1] - t[0]
        except:
            dt=0
        # if dt==0,inf will appear in voxelgrid
        if len(x) != len(y) or len(y) != len(t) or len(t) != len(p) or len(x) == 0 or dt == 0:
            print("empty or abnormal event sequence:")
            return torch.zeros(num_bins, H, W)
        t = t - start_time
        if reverse:
            t = end_time - t
            p = -p
            x = numpy.flipud(x)
            y = numpy.flipud(y)
            t = numpy.flipud(t)
            p = numpy.flipud(p)

        t_norm = (t - t[0]) / dt
        voxel_grid = get_voxel_grid(torch.from_numpy(x.copy().astype(numpy.float32)),
                                    torch.from_numpy(y.copy().astype(numpy.float32)),
                                    torch.from_numpy(t_norm.copy().astype(numpy.float32)),
                                    torch.from_numpy(p.copy().astype(numpy.float32)), num_bins, H, W)
        return voxel_grid

    def __len__(self):
        return len(self.data_dic)

    def splite_dataset(self):
        h5_paths = sorted(glob.glob(os.path.join(self.root, "*.h5")))
        print(h5_paths)
        for h5_path in h5_paths:
            imgs = []
            time_stamp_list = []
            event_index_list = []
            f = h5py.File(h5_path, 'r')
            event_sequence = numpy.stack((f["events/xs"], f["events/ys"], f["events/ts"], f["events/ps"]), axis=-1)
            for img_dataset in f["images"]:
                # print(f["images"][img_dataset].attrs.keys())
                img = f["images"][img_dataset][:]
                imgs.append(img)
                time_stamp_list.append(f["images"][img_dataset].attrs["timestamp"])
                event_index_list.append(f["images"][img_dataset].attrs["event_idx"])
            for i in range(len(imgs)):
                if i % (self.skip_frames + 1) == 0 and (i + self.skip_frames + 1) < len(f["images"]):
                    start_img_index = i
                    end_img_index = start_img_index + self.skip_frames + 1
                    inter_events = event_sequence[event_index_list[start_img_index]+1:event_index_list[end_img_index], :]
                    timestamp_list = [time_stamp_list[start_img_index],
                                      time_stamp_list[end_img_index]]
                    data = {"start_image": imgs[start_img_index],
                            "end_image": imgs[end_img_index],
                            "mid": [],
                            "timestamps": timestamp_list,
                            "event_sequences": inter_events
                            }

                    if self.label_require:
                        for j in range(self.num_inter):
                            true_img_index = start_img_index + j + 1
                            true_img = imgs[true_img_index]
                            data["mid"].append(
                                {"true_image": true_img,
                                 "mid_timestamp": time_stamp_list[true_img_index]})
                    self.data_dic.append(data)

