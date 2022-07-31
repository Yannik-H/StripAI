from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class StripDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.image_ids = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        label2int = {"CE": 0, "LAA": 1}

        image_id = self.image_ids[idx]
        image_dir = "../data/train_tiles/all_imgs/"
        images = []
        for i in range(self.cfg["n_images"]):
            path = image_dir + image_id + f'_{i}.jpg'
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)["image"]
            images.append(torch.tensor(image))
        images = torch.cat(images, dim=2)   # There is a depth dimension originally, I cat the dimension with the RGB channel
        images = images.reshape([48, 1024, 1024])
        label = torch.tensor(label2int[self.labels[idx]])#.long()
        return images, label