import os, numpy
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class RESIDE_Dataset(data.Dataset):
    def __init__(self, dataset_path, transform=None):
        super(RESIDE_Dataset, self).__init__()
        self.transform = transform
        self.clear_dir = os.path.join(dataset_path, 'clear')
        self.haze_dir = os.path.join(dataset_path, 'hazy')
        self.img_list = os.listdir(self.haze_dir)

    def __getitem__(self, index):
        haze_img_url = os.path.join(self.haze_dir, self.img_list[index])
        clear_img_url = os.path.join(self.clear_dir, self.img_list[index])

        haze = Image.open(haze_img_url)
        clear = Image.open(clear_img_url)

        haze, clear =haze.convert("RGB"), clear.convert("RGB")

        if self.transform:
            haze, clear = numpy.array(haze), numpy.array(clear)
            imgs = self.transform(image=haze, mask=clear)
            haze = imgs["image"]
            clear = imgs["mask"]
            haze, clear = Image.fromarray(haze), Image.fromarray(clear)
        haze = transforms.ToTensor()(haze)
        clear = transforms.ToTensor()(clear)
        return haze, clear

    def augData(self, data, target):
        data = transforms.ToTensor()(data)
        data = transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = transforms.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.img_list)

if __name__ == "__main__":
    pass
