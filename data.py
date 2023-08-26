from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import os
from glob import glob
from natsort import natsorted
import cv2
from hypr import config


def data_preprocess(img):
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    img = img / 255.0
    return img


class SR2k(Dataset):
    def __init__(self, config, set_type):
        self.device = config["device"]
        # get the image path list -> self.image_names
        self.HR_names = natsorted(glob(os.path.join(config["data_dir"], "HR", "*.png")))
        self.LR_names = natsorted(glob(os.path.join(config["data_dir"], "LR", "*.png")))

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
        HR_img = data_preprocess(cv2.imread(HR_name))
        LR_name = self.LR_names[idx]
        LR_img = data_preprocess(cv2.imread(LR_name))

        return {"LR": LR_img, "HR": HR_img}


def main():
    train_ds = SR2k(config, set_type="train")
    val_ds = SR2k(config, set_type="val")
    train_dl = DataLoader(train_ds, config["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, config["batch_size"], shuffle=True, drop_last=True)


if __name__ == "__main__":
    main()
