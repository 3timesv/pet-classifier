import yaml

import albumentations as A
from albumentations.pytorch import ToTensorV2

with open("config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)


class CustomTransform:
    def __init__(self, image_transform, label_transform):
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __call__(self, sample):
        image, label = sample

        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        if self.label_transform is not None:
            label = self.label_transform(label)

        return (image, label)


class PetTransform:
    train_image_transform = A.Compose([
        A.Resize(height=config["input_size"]["presize_height"],
                 width=config["input_size"]["presize_width"]),
        A.RandomResizedCrop(height=config["input_size"]["final_height"],
                            width=config["input_size"]["final_width"],
                            ratio=(1, 1)),
        A.Normalize(),
        ToTensorV2()])

    valid_image_transform = A.Compose([
        A.Resize(height=config["input_size"]["final_height"],
                 width=config["input_size"]["final_width"]),
        A.Normalize(),
        ToTensorV2()])

    train = CustomTransform(train_image_transform, None)
    valid = CustomTransform(valid_image_transform, None)
