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

    model.model.eval()

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
        model.model.cuda()
        model.model.train()
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


def test(ratings):

    l = len(ratings)
    ans = 0
    left, right = 0, 1

    while right < l:
        if ratings[right] - ratings[right - 1] == 1:
            right += 1
        else:
            diff = right - left
            ans += (diff + 1) * diff / 2
            left = right
            right += 1

    return ans


if __name__ == "__main__":

    test([])

    train_csv = pd.read_csv("../data/train.csv")
    train, val = train_test_split(train_csv, test_size=0.2, random_state=66)

    train_dataset = StripDataset({"n_images": 16}, train)
    val_dataset = StripDataset({"n_images": 16}, val)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    model = StripEfficientNet("efficientnet-b4", False, None, in_channels=48, num_classes=2)
    writer = SummaryWriter("../train_log/efficientnet_b4")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

    train_model(model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=None,
                num_epochs=1,
                log_writer=writer)

    print("Finished")

    # pip3 install torch torchvision torchaudio
