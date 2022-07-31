from torch import nn
from torch.utils.data import Dataset, DataLoader
from dataset import StripDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch
from torch.utils.tensorboard import SummaryWriter
from efficientnet import StripEfficientNet


def val_model(model, val_dataloader):
    pass


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs, log_writer=None):

    for epoch in range(num_epochs):
        model.model.cuda()
        step = 0
        for x, y in tqdm(train_dataloader):
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
                    log_writer.add_scalar(f"epoch{epoch} loss", loss, global_step=step)
                else:
                    print(loss)

            step += 1



if __name__ == "__main__":

    train_csv = pd.read_csv("../data/train.csv")
    train, val = train_test_split(train_csv, test_size=0.2, random_state=66)

    train_dataset = StripDataset({"n_images":16}, train)
    val_dataset = StripDataset({"n_images":16}, val)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    model = StripEfficientNet("efficientnet-b4", False, None, in_channels=48, num_classes=2)
    # writer = SummaryWriter("../train_log/efficientnet_b4")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    train_model(model, criterion, optimizer, train_dataloader, None, 1, None)

    print("Finished")

    # pip3 install torch torchvision torchaudio