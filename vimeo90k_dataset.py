import numpy
import os
import torch
import torch.utils.data
from tqdm import tqdm
from PIL import Image
from utils.event_utils import events_to_voxel_torch,events_to_timestamp_image,events_to_image_torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import sys
import glob
from torchvision.transforms import ToPILImage

# return (num_bins,H,W)
def get_voxel_grid(xs, ys, ts, ps, num_bins, H, W):
    # hot_events_mask = numpy.ones([num_bins, H,W])
    # hot_events_mask = torch.from_numpy(hot_events_mask).float()
    # generate voxel grid which has size self.num_bins x H x W
    voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, num_bins, sensor_size=(H, W))
    # print('voxel',voxel_grid.size())
    # voxel_grid = voxel_grid * hot_events_mask
    return voxel_grid


class Vimeo90kDataset():
    def __init__(self, tri_root, sep_root, img_size=None, mode='septuplet_train', num_bins=5, n_inputs=2, skip=1):
        self.num_bins = num_bins
        self.img_size = img_size
        self.tri_root = tri_root
        self.sep_root = sep_root
        self.skip = skip
        self.data_dic = []
        self.mode = mode
        self.n_inputs = n_inputs

        self.splite_dataset()

    def __getitem__(self, index):
        events_t_0 = []
        events_t_1 = []
        events_0_1 = []
        events_1_0 = []
        data = self.data_dic[index]

        true_images = [transforms.ToTensor()(Image.open(image)) for image in data["true_images"]]
        input_images = [transforms.ToTensor()(Image.open(image)) for image in data["input_images"]]
        (H, W) = true_images[0].shape[-2], true_images[0].shape[-1]
        # print(index, "   ",index + self.num_inter + 1)
        for inter_index in range(self.skip):
            event_t_0, event_t_1, event_0_1, event_1_0  = self.load_events_data(data["before_events"][inter_index],
                                                                           data["after_events"][inter_index],
                                                                           H, W)
            events_t_0.append(event_t_0)
            events_t_1.append(event_t_1)
            events_0_1.append(event_0_1)
            events_1_0.append(event_1_0)
        return input_images, true_images, events_t_0, events_t_1, events_0_1[0],events_1_0[0]

    def load_events_data(self, before_event_files, after_event_files, H, W):
        x, y, t, p = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        total_files = before_event_files + after_event_files
        num_events_0_t = 0
        for i in range(len(total_files)):
            event_file = total_files[i]
            try:
                event_data = numpy.load(event_file)
            except:
                continue
            if len(event_data["arr_0"][:, 2]) > 1:
                event_data["arr_0"][:, 2][-1] += 1
                x = torch.cat([x, torch.from_numpy(event_data["arr_0"][:, 0].astype(numpy.float32).reshape((-1,)))], -1)
                y = torch.cat([y, torch.from_numpy(event_data["arr_0"][:, 1].astype(numpy.float32).reshape((-1,)))], -1)
                t = torch.cat([t, torch.from_numpy(event_data["arr_0"][:, 2].astype(numpy.float32).reshape((-1,)))], -1)
                p = torch.cat([p, torch.from_numpy(event_data["arr_0"][:, 3].astype(numpy.float32).reshape((-1,)))], -1)
            if i == (len(before_event_files) - 1):
                num_events_0_t = len(x)

        if len(t) > 0:
            mask = (x <= W - 1) & (y <= H - 1) & (x >= 0) & (y >= 0)
            x = x[mask]
            y = y[mask]
            p = p[mask]
            t = t[mask]

            reverse_t = t.max() + t.min() - t
            reverse_t =torch.flip(reverse_t, [0])
            reverse_x =torch.flip(x, [0])
            reverse_y = torch.flip(y, [0])
            reverse_p = -torch.flip(p, [0])

            before_t = t[:num_events_0_t]
            before_t = before_t.max() + before_t.min() - before_t
            before_t = torch.flip(before_t, [0])
            before_x = x[:num_events_0_t]
            before_x = torch.flip(before_x, [0])
            before_y = y[:num_events_0_t]
            before_y = torch.flip(before_y, [0])
            before_p = p[:num_events_0_t]
            before_p = -torch.flip(before_p, [0])

            event_0_1 = get_voxel_grid(x, y, t, p, 2*self.num_bins, H, W)
            event_1_0 = get_voxel_grid(reverse_x, reverse_y, reverse_t, reverse_p, 2 * self.num_bins, H, W)
            if len(before_t)>0:
                event_t_0 = get_voxel_grid(before_x,before_y, before_t, before_p, self.num_bins, H, W)
            else:
                event_t_0 = torch.zeros(self.num_bins, H, W)
            if len(t)>len(before_t):
                event_t_1 = get_voxel_grid(x[num_events_0_t:], y[num_events_0_t:], t[num_events_0_t:], p[num_events_0_t:], self.num_bins, H, W)
            else:
                event_t_1 = torch.zeros(self.num_bins, H, W)

        else:
            event_t_0 = torch.zeros(self.num_bins, H, W)
            event_t_1 = torch.zeros(self.num_bins, H, W)
            event_0_1 = torch.zeros(2 * self.num_bins, H, W)
            event_1_0= torch.zeros(2 * self.num_bins, H, W)
        return event_t_0, event_t_1, event_0_1, event_1_0

    def __len__(self):
        return len(self.data_dic)

    def splite_dataset(self):
        if self.mode == 'triplet_train' or self.mode == 'triplet_test':
            if self.mode == 'triplet_train':
                list_path = os.path.join(self.tri_root, "tri_trainlist.txt")
            else:
                list_path = os.path.join(self.tri_root, "tri_testlist.txt")
            with open(list_path, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline().strip('\n')  # 整行读取数据,去掉换行付
                    # lines.strip()遇空白行返回FALSE
                    if not lines or not lines.strip():
                        break

                    after_event_path = []
                    before_event_path = []
                    true_images_path = []
                    input_images_path = []
                    for i in range(1, 4):
                        if i >= (2 - int(self.skip / 2)) and i <= (2 + int(self.skip / 2)):
                            true_images_path.append(os.path.join(self.tri_root, "sequences", lines, "im%d.png" % i))
                            inter_before_events = []
                            inter_after_events = []

                            for before_event_index in range(int(2 - self.n_inputs / 2 - int(self.skip / 2)) - 1, i - 1):
                                inter_before_events.append(
                                    os.path.join(self.tri_root, "events", lines, "%010d.npz" % before_event_index))
                            for after_event_index in range(i - 1, int(2 + self.n_inputs / 2 + int(self.skip / 2)) - 1):
                                inter_after_events.append(
                                    os.path.join(self.tri_root, "events", lines, "%010d.npz" % after_event_index))
                            before_event_path.append(inter_before_events)
                            after_event_path.append(inter_after_events)
                        elif i >= (2 - self.n_inputs / 2 - int(self.skip / 2)) and i <= (
                                2 + self.n_inputs / 2 + int(self.skip / 2)):
                            input_images_path.append(os.path.join(self.tri_root, "sequences", lines, "im%d.png" % i))

                    if not os.path.exists(before_event_path[-1][-1]):
                        continue
                    data = {'input_images': input_images_path, 'true_images': true_images_path,
                            'before_events': before_event_path, 'after_events': after_event_path}
                    # print(data)
                    self.data_dic.append(data)
        elif self.mode == 'septuplet_train' or self.mode == 'septuplet_test':
            if self.mode == 'septuplet_train':
                list_path = os.path.join(self.sep_root, "sep_trainlist.txt")
            else:
                list_path = os.path.join(self.sep_root, "sep_testlist.txt")

            with open(list_path, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline().strip('\n')  # 整行读取数据,去掉换行付
                    # lines.strip()遇空白行返回FALSE
                    if not lines or not lines.strip():
                        break

                    after_event_path = []
                    before_event_path = []
                    true_images_path = []
                    input_images_path = []
                    for i in range(1, 8):
                        if i >= (4 - int(self.skip / 2)) and i <= (4 + int(self.skip / 2)):
                            true_images_path.append(os.path.join(self.sep_root, "sequences", lines, "im%d.png" % i))
                            inter_before_events = []
                            inter_after_events = []
                            for before_event_index in range(int(4 - self.n_inputs / 2 - int(self.skip / 2)), i):
                                inter_before_events.append(
                                    os.path.join(self.sep_root, "events", lines, "%010d.npz" % before_event_index))
                            for after_event_index in range(i, int(4 + self.n_inputs / 2 + int(self.skip / 2))):
                                inter_after_events.append(
                                    os.path.join(self.sep_root, "events", lines, "%010d.npz" % after_event_index))

                            before_event_path.append(inter_before_events)
                            after_event_path.append(inter_after_events)
                        elif i >= (4 - self.n_inputs / 2 - int(self.skip / 2)) and i <= (
                                4 + self.n_inputs / 2 + int(self.skip / 2)):
                            input_images_path.append(os.path.join(self.sep_root, "sequences", lines, "im%d.png" % i))

                    if not os.path.exists(before_event_path[-1][-1]):
                        continue
                    data = {'input_images': input_images_path, 'true_images': true_images_path,
                            'before_events': before_event_path, 'after_events': after_event_path}
                    # print(data)
                    self.data_dic.append(data)
        else:
            print('this mode is not exist:', self.mode)
            return 1


if __name__ == '__main__':
    train_dataset = VimeoDataset(tri_root="D:\\LHX\\CVPR\\dataset\\vimeo_triplet",
                                 sep_root="D:\\LHX\\CVPR\\dataset\\vimeo_septuplet",
                                 num_bins=5,
                                 mode='septuplet_test',
                                 n_inputs=6,
                                 skip=1)
    # train_dataset = VimeoDataset(tri_root="/gs/home/jinjing/NewEventTransformer/vimeo_triplet",
    #                              sep_root="/gs/home/jinjing/NewEventTransformer/vimeo_septuplet",
    #                              num_bins=5,
    #                              mode='triplet_train',
    #                               n_inputs=2,
    #                               skip=1)
