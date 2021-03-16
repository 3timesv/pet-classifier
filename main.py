import os
import random
from tqdm import tqdm
import albumentations as A
import dataset
import models
import timm
import torch
import torch.nn as nn
import utils
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

IMAGE_PATH = "data/images/"
TRAIN_SPLIT = 0.8
HEIGHT, WIDTH = 128, 128
NUM_CLASSES = 37
PARAMS = {
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "lr": 0.001,
    "batch_size": 8,
    "epochs": 10
}

fpaths = [os.path.join(IMAGE_PATH, fname) for fname in os.listdir(IMAGE_PATH)]
random.shuffle(fpaths)

train_list = fpaths[:int(len(fpaths) * TRAIN_SPLIT)]
valid_list = fpaths[int(len(fpaths) * TRAIN_SPLIT):]

train_image_transform = A.Compose([
    A.Resize(height=HEIGHT, width=WIDTH),
    ToTensorV2()])

train_transform = dataset.PetTransform(train_image_transform, None)

train_ds = dataset.PetDataset(train_list, utils.get_label, train_transform)
valid_ds = dataset.PetDataset(valid_list, utils.get_label, train_transform)

train_loader = DataLoader(train_ds,
                          batch_size=PARAMS["batch_size"],
                          shuffle=True)

valid_loader = DataLoader(valid_ds,
                          batch_size=PARAMS["batch_size"],
                          shuffle=False)

model = models.CustomResnet(model_name='resnet34',
                            pretrained=True,
                            target_size=NUM_CLASSES).to(PARAMS["device"])

criterion = nn.CrossEntropyLoss().to(PARAMS["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])

print("Starting Training with", PARAMS["device"])

for epoch in range(1, PARAMS["epochs"]+1):
    sum_loss = 0.0
    stream = tqdm(train_loader)
    for i, sample in enumerate(stream):
        images = sample["image"].to(PARAMS["device"]).float()
        target = sample["label"].to(PARAMS["device"])

        output = model(images)

        loss = criterion(output, target)
        sum_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch", epoch)
    print("Training loss:", sum_loss / i)

    model.eval()
    sum_loss = 0.0
    stream = tqdm(valid_loader)
    for i, sample in enumerate(stream):
        images = sample["images"].to(PARAMS["device"]).float()
        target = sample["label"].to(PARAMS["device"])

        output = model(images)
        loss = criterion(output, target)

        sum_loss += loss.item()

    print("Valid loss:", sum_loss / i)

