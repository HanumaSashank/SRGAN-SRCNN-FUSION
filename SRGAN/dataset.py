import os
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class MyDataFolder(Dataset):
    def __init__(self, root_dir):
        super(MyDataFolder, self).__init__()
        self.img_data = []
        self.root_dir = root_dir
        self.class_label = os.listdir(root_dir)

        for idx, name in enumerate(self.class_label):
            files_list = os.listdir(os.path.join(root_dir, name))
            self.img_data += list(zip(files_list, [idx] * len(files_list)))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img_file, label = self.img_data[index]
        root_n_dir = os.path.join(self.root_dir, self.class_label[label])

        image = np.array(Image.open(os.path.join(root_n_dir, img_file)))
        image = config.both_resolution_transforms(image=image)["image"]
        high_res_img = config.high_resolution_transform(image=image)["image"]
        low_res_img = config.low_resolution_transform(image=image)["image"]
        return low_res_img, high_res_img


def test():
    image_dataset = MyDataFolder(root_dir="img_data/")
    loader = DataLoader(image_dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
