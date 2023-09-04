from torch.utils.data import Dataset
# import torchvision.transforms as T
import torchvision.transforms.functional as FT
import torch
import os
from glob import glob
from natsort import natsorted
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from hypr import config
import random


def transform(HR_img, LR_img):
    HR_img = np.transpose(HR_img if HR_img.shape[2] == 1 else HR_img[:, :, [
                          2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
    LR_img = np.transpose(LR_img if LR_img.shape[2] == 1 else LR_img[:, :, [
                          2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
    HR_img = torch.from_numpy(HR_img).float()
    LR_img = torch.from_numpy(LR_img).float()
    # crop_size = 96
    # i, j, h, w = T.RandomCrop.get_params(LR_img, (crop_size, crop_size))
    # HR_img = FT.crop(HR_img, i*3, j*3, h*3, w*3)
    # LR_img = FT.crop(LR_img, i, j, h, w)
    if random.random() < 0.5:
        HR_img = FT.hflip(HR_img)
        LR_img = FT.hflip(LR_img)
    if random.random() < 0.5:
        HR_img = FT.vflip(HR_img)
        LR_img = FT.vflip(LR_img)
    return HR_img, LR_img


class SR2k(Dataset):
    def __init__(self, config, set_type):
        self.device = config["device"]
        # get the image path list -> self.image_names
        self.HR_names = natsorted(
            glob(os.path.join(config["data_dir"], "HR", "*.png")))
        self.LR_names = natsorted(
            glob(os.path.join(config["data_dir"], "LR", "*.png")))

        if set_type == "train":
            n_start = 0
            n_end = config["data_split_num"]
        elif set_type == "val":
            n_start = config["data_split_num"]
            n_end = config["data_num"]

        self.HR_names = self.HR_names[n_start:n_end]
        self.LR_names = self.LR_names[n_start:n_end]

    def __len__(self):
        return len(self.HR_names)

    def __getitem__(self, idx):
        HR_name = self.HR_names[idx]
        HR_img = cv.imread(HR_name).astype(np.float32) / 255
        LR_name = self.LR_names[idx]
        LR_img = cv.imread(LR_name).astype(np.float32) / 255
        HR_img, LR_img = transform(HR_img, LR_img)
        return {"LR": LR_img, "HR": HR_img}


def main():
    HR_names = natsorted(glob(os.path.join(config["data_dir"], "HR", "*.png")))
    LR_names = natsorted(glob(os.path.join(config["data_dir"], "LR", "*.png")))
    HR_img = cv.imread(HR_names[0]).astype(np.float32) / 255
    LR_img = cv.imread(LR_names[0]).astype(np.float32) / 255
    HR_img, LR_img = transform(HR_img, LR_img)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(HR_img.permute(1, 2, 0))
    axarr[1].imshow(LR_img.permute(1, 2, 0))
    # plt.savefig("plt.png")
    plt.show()


if __name__ == "__main__":
    main()
