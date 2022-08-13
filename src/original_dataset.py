from torch.utils.data import Dataset, DataLoader
import cv2

class ImgDataset(Dataset):

    def __init__(self, df, img_dir):
        self.df = df
        self.train = 'label' in df.columns
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        paths = ["../input/jpg-images-strip-ai/test/", "../input/jpg-images-strip-ai/train/"]
        image = cv2.imread(self.img_dir + self.df.iloc[index].image_id + ".jpg")
        if len(image.shape) == 5:
            image = image.squeeze().transpose(1, 2, 0)
        image = cv2.resize(image, (512, 512)).transpose(2, 0, 1)
        label = None
        if(self.train):
            label = {"CE" : 0, "LAA": 1}[self.df.iloc[index].label]

        return image, label