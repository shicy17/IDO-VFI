import numpy
import os
import torch
import torch.utils.data
from utils.event_utils import events_to_voxel_torch
from PIL import Image
import torchvision.transforms as transforms
import glob
from torchvision.transforms import ToPILImage


# return (num_bins,H,W)
def get_voxel_grid(xs, ys, ts, ps, num_bins, H, W):
    # generate voxel grid which has size self.num_bins x H x W
    voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(H, W))
    return voxel_grid


class RunDataset():
    def __init__(self, root, num_bins=5, skip_frames=1, num_inter=1):
        self.num_bins = num_bins
        self.root = root
        self.skip_frames = skip_frames
        self.num_inter = num_inter
        self.data_dic = []
        self.split_dataset()

    def __getitem__(self, index):
        data = self.data_dic[index]
        I0 = Image.open(data["start_image"])
        I1 = Image.open(data["end_image"])
        I0 = transforms.ToTensor()(I0)
        I1 = transforms.ToTensor()(I1)
        input_images = [I0, I1]
        start_time = data["timestamps"][0]
        end_time = data["timestamps"][1]
        (H, W) = (I0.shape[-2], I1.shape[-1])
        event_0_1 = self.load_events_data(data["event_files"], H, W)
        voxel_grid_0_1 = self.event_sequence_to_voxel_grid(event_0_1, start_time, end_time, 2 * self.num_bins, H, W,
                                                           reverse=False)
        voxel_grid_1_0 = self.event_sequence_to_voxel_grid(event_0_1, start_time, end_time, 2 * self.num_bins, H, W,
                                                           reverse=True)
        voxel_grids_t_0 = []
        voxel_grids_t_1 = []
        mid_timestamps = numpy.linspace(start_time, end_time, self.num_inter + 2)[1:-1]
        for inter_index in range(self.num_inter):
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

        return input_images, voxel_grids_t_0, voxel_grids_t_1, voxel_grid_0_1, voxel_grid_1_0

    def event_sequence_to_voxel_grid(self, event_sequence, start_time, end_time, num_bins, H, W, reverse=False):
        x = event_sequence[:, 0]
        y = event_sequence[:, 1]
        t = event_sequence[:, 2]
        p = event_sequence[:, 3]
        try:
            dt = t[-1] - t[0]
        except:
            dt = 0
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
        voxel_grid = get_voxel_grid(torch.from_numpy(x.copy().astype(numpy.float32)),
                                    torch.from_numpy(y.copy().astype(numpy.float32)),
                                    torch.from_numpy(t.copy().astype(numpy.float32)),
                                    torch.from_numpy(p.copy().astype(numpy.float32)), num_bins, H, W)
        return voxel_grid

    def load_events_data(self, event_files, H, W):
        x, y, t, p = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        for event_file in event_files:
            try:
                event_data = numpy.load(event_file)
            except:
                print("can't load file:", event_file)
                continue
            x = numpy.hstack((x, event_data["x"].astype(numpy.float32).reshape((-1,))))
            y = numpy.hstack((y, event_data["y"].astype(numpy.float32).reshape((-1,))))
            t = numpy.hstack((t, event_data["t"].astype(numpy.float32).reshape((-1,))))
            p = numpy.hstack((p, event_data["p"].astype(numpy.float32).reshape((-1,)) * 2 - 1))
        mask = (x <= W - 1) & (y <= H - 1) & (x >= 0) & (y >= 0)
        x_ = x[mask]
        y_ = y[mask]
        p_ = p[mask]
        t_ = t[mask]
        if len(x_) != len(y_) or len(y_) != len(t_) or len(t_) != len(p_) or len(x_) == 0:
            print("empty or abnormal event sequence:", event_files)
            x_, y_, t_, p_ = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        event_sequence = numpy.stack((x_, y_, t_, p_), axis=-1)
        return event_sequence

    def __len__(self):
        return len(self.data_dic)

    def split_dataset(self):
        # n.npz contains events from n-1.png to n.png
        event_root = os.path.join(self.root, "events_aligned")
        img_root = os.path.join(self.root, "images_corrected")
        time_stamp_path = os.path.join(self.root, 'images_corrected', 'timestamp.txt')
        image_files = sorted(glob.glob(os.path.join(img_root, "*.png")))
        event_files = sorted(glob.glob(os.path.join(event_root, "*.npz")))

        f = open(time_stamp_path)
        time_stamp_list = f.readlines()
        print(dir, "num time stamp", len(time_stamp_list), "num event files", len(event_files))
        for i in range(len(time_stamp_list)):
            time_stamp_list[i] = time_stamp_list[i].replace("\n", " ")
            time_stamp_list[i] = float(time_stamp_list[i].strip("\n"))
        f.close()
        for i in range(len(image_files)):
            if i % (self.skip_frames + 1) == 0:
                start_img_index = i
                end_img_index = start_img_index + self.skip_frames + 1
                inter_events = []

                if end_img_index < len(image_files) and end_img_index < len(event_files):
                    for index_event in range(start_img_index, end_img_index):
                        inter_events.append(event_files[index_event])

                    timestamp_list = [time_stamp_list[start_img_index],
                                      time_stamp_list[end_img_index]]
                    data = {"start_image": image_files[start_img_index],
                            "end_image": image_files[end_img_index],
                            "timestamps": timestamp_list,
                            "event_files": inter_events
                            }
                    self.data_dic.append(data)
            else:
                continue

