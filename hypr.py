import torch

config = {
    "data_dir": "./data/sorted",
    "data_split_num": 800,
    "data_num": 900,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # you can set your own training configurations
    "batch_size": 1,
    "learning_rate": 1e-3,
    "epoch_num": 30,
    "resume": False,
}
