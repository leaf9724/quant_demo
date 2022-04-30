import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class MyDataSet(Dataset):
    def __init__(self, file_path, transforms=None):
        super(MyDataSet, self).__init__()
        path_list = os.listdir(file_path)
        # path_list.sort(key=lambda x:int(x[:-4])
        self.root = file_path
        self.path_list = path_list
        #open(file_name_path).readlines()
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path, label = self.path_list[idx], self.path_list[idx][-7]
        print
        img = Image.open(os.path.join(self.root,img_path))  # numpy array
        if self.transforms is not None:
            img = self.transforms(img)
        return img, int(label)

    def __len__(self):
        return len(self.path_list)


# mydataset = MyDataSet(r'/home/pc/matrad/leaf/factor/daily_data/data_processed/greay_picture')
# vailddataset = MyDataSet(r'/home/pc/matrad/leaf/factor/daily_data/data_processed/grey_vaild')
# mydataloader = DataLoader(mydataset, batch_size=64, drop_last=False)