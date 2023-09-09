import torch

config = {
    "data_dir": "./data/sorted",
    "data_split_num": 3300,
    "data_num": 3550,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1,
    "learning_rate": 1e-3,
    "epoch_num": 50,
    "resume": False,
}
