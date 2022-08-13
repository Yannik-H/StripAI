import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MIL import SimpleMIL
from dataset import StripDataset
from original_dataset import ImgDataset
from CoAtNet import CoAtNet


def val_model(model, val_dataloader):
    model.eval()

    val_loss = .0
    val_acc = .0

    iterator = tqdm(val_dataloader)

    iterator.set_description("Validating")

    for x, y in iterator:
        x = x.cuda().float()
        y = y.cuda().long()

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            output = model.forward(x)
            loss = criterion(output, y)
            _, preds = torch.max(output, 1)

            val_loss += loss
            val_acc += torch.sum(preds == y.data)

    data_size = len(val_dataloader.dataset)

    return val_loss / data_size, val_acc / data_size


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs, log_writer=None):
    for epoch in range(num_epochs):
        model.cuda()
        model.train()
        step = 0

        iterator = tqdm(train_dataloader)
        for x, y in iterator:
            x = x.cuda().float()
            y = y.cuda().long()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model.forward(x)
                loss = criterion(output, y)
                _, preds = torch.max(output, 1)

                loss.backward()
                optimizer.step()

                if log_writer:
                    log_writer.add_scalar(f"epoch{epoch} train loss", loss, global_step=step)

            iterator.set_description(f"Epoch{epoch} train loss: {float(loss.data)}")

            step += 1

        val_loss, acc = val_model(model, val_dataloader)
        log_writer.add_scalar(f"val loss", val_loss, global_step=epoch)
        log_writer.add_scalar(f"val acc", acc, global_step=epoch)


if __name__ == "__main__":
    train_csv = pd.read_csv("../data/train.csv")
    train, val = train_test_split(train_csv, test_size=0.2, random_state=66)

    # train_dataset = StripDataset({"n_images": 16}, train)
    # val_dataset = StripDataset({"n_images": 16}, val)
    train_dataset = ImgDataset(train, "../data/train_tiles/train_jpg/")
    val_dataset = ImgDataset(val, "../data/train_tiles/train_jpg/")

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)
    # model = StripEfficientNet("efficientnet-b4", False, None, in_channels=48, num_classes=2)
    # model = SimpleMIL(model_name="tf_efficientnet_b4_ns")
    num_blocks = [2, 2, 12, 28, 2]
    channels = [64, 64, 128, 256, 512]
    model = CoAtNet((512, 512), 3, num_blocks, channels, num_classes=2)

    writer = SummaryWriter("../train_log/CoAtNet")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_model(model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                num_epochs=5,
                log_writer=writer)

    print("Finished")

    # pip3 install torch torchvision torchaudio
