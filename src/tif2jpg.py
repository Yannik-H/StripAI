import pandas as pd
import cv2
from tqdm import tqdm
import tifffile

train_df = pd.read_csv("../data/train.csv")

# max_count = max(train_df.label.value_counts())
# for label in train_df.label.unique():
#     df = train_df.loc[train_df.label == label]
#     while(train_df.label.value_counts()[label] < max_count):
#         train_df = pd.concat([train_df, df.head(max_count - train_df.label.value_counts()[label])], axis = 0)


for i in tqdm(range(train_df.shape[0])):
        img_id = train_df.iloc[i].image_id
        img = cv2.resize(tifffile.imread("../data/train_tiles/train/" + img_id + ".tif"), (512, 512))
        cv2.imwrite(f"../data/train_tiles/train_jpg/{img_id}.jpg", img)

print("Finished!")
