from torch.utils.data import Dataset
import cv2
import re
import os


class PetDataset(Dataset):
    def __init__(self, fpaths, label_func, transform=None):

        self.fpaths = fpaths
        self.label_func = label_func
        self.transform = transform

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):

        fpath = self.fpaths[idx]

        image = cv2.imread(fpath)

        label = self.label_func(fpath.split('/')[-1])

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PetTransform:
    def __init__(self, image_transform, label_transform):
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if self.image_transform is not None:
            image = self.image_transform(image=sample["image"])["image"]

        if self.label_transform is not None:
            label = self.label_transform(sample["label"])

        return {"image": image,
                "label": label}

        
