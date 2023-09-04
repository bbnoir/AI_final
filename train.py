from hypr import config
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import SR2k
from model import SuperResolution
import numpy as np
from tqdm import tqdm

from my_demo import demo


def main():
    # data
    train_ds = SR2k(config, set_type="train")
    val_ds = SR2k(config, set_type="val")
    train_dl = DataLoader(
        train_ds, config["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(
        val_ds, config["batch_size"], shuffle=True, drop_last=True)

    device = config["device"]

    model = SuperResolution().to(device)

    criterion = nn.L1Loss()

    n_epochs = config["epoch_num"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=2, verbose=False
    )

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    # initialize history for recording what we want to know
    history = []
    start_epoch = 0

    # training resume
    if config["resume"]:
        checkpoint = torch.load("./checkpoint.pth")
        print("Loading checkpoint model...")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loading checkpoint optimizer...")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        valid_loss_min = checkpoint["valid_loss_min"]
        start_epoch = checkpoint["epoch"]
        print(
            "Start at epoch",
            start_epoch,
            "with the min valid loss: " + str(valid_loss_min),
        )

    for epoch in range(start_epoch, n_epochs):
        # monitor training loss, validation loss and learning rate
        train_loss = 0.0
        valid_loss = 0.0
        lrs = []
        result = {"train_loss": [], "val_loss": [], "lrs": []}

        # train the model
        model.train()
        for data_pack in tqdm(train_dl):
            data = data_pack["LR"].to(device)
            target = data_pack["HR"].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # record learning rate
            lrs.append(optimizer.param_groups[0]["lr"])

            # record loss
            train_loss += loss.item() * data.size(0)

        # validate the model
        model.eval()
        for data_pack in val_dl:
            data = data_pack["LR"].to(device)
            target = data_pack["HR"].to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        scheduler.step(valid_loss)
        # scheduler.step()

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_dl.dataset)
        result["train_loss"] = train_loss
        valid_loss = valid_loss / len(val_dl.dataset)
        result["val_loss"] = valid_loss
        leaning_rate = lrs
        result["lrs"] = leaning_rate

        print(
            "Epoch {:2d}: LR: {:.6f} Train Loss: {:.6f} Val Loss:{:.6f}"
            .format(
                epoch + 1, leaning_rate[-1], train_loss, valid_loss
            )
        )

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                "Val loss decreased({:.6f}-->{:.6f}). => Saving checkpoint"
                .format(
                    valid_loss_min, valid_loss
                )
            )
            valid_loss_min = valid_loss
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epich": epoch,
                "valid_loss_min": valid_loss_min,
            }
            torch.save(state, "checkpoint.pth")
            torch.save(model.state_dict(), "model.pth")

        psnr = demo()
        result["psnr"] = psnr
        history.append(result)
        print(f'Average PSNR: {psnr:.3f} dB')

    plot_lrs(history)
    plot_losses(history)
    plot_psnrs(history)


def train():
    pass


def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.figure()
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.savefig("plot/lr.png")


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig("plot/loss.png")


def plot_psnrs(history):
    psnrs = [x.get('psnrs') for x in history]
    plt.figure()
    plt.plot(psnrs)
    plt.xlabel('epoch')
    plt.ylabel('psnrs')
    plt.title('PSNR vs. No. of epochs')
    plt.savefig("plot/psnr.png")


if __name__ == "__main__":
    main()
