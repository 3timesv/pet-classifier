import os
import random
import sys
from pathlib import Path

import yaml

import torch
import torch.nn as nn
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
import utils
from dataset import Loader
from models import CustomResnet
from preprocess import PetTransform
from trainer import Trainer



with open("config.yaml", 'r') as config_file:
    CONFIG = yaml.safe_load(config_file)

path_list = list(Path(CONFIG["image_path"]).glob("*.jpg"))
train_path_list, valid_path_list = utils.split(
    path_list, CONFIG["valid_split"])


train_loader = Loader(path_list=train_path_list,
                      label_func=utils.get_label,
                      batch_size=CONFIG["batch_size"],
                      shuffle=True,
                      transform=PetTransform.train)

valid_loader = Loader(path_list=valid_path_list,
                      label_func=utils.get_label,
                      batch_size=CONFIG["batch_size"],
                      shuffle=False,
                      transform=PetTransform.valid)

model = CustomResnet(CONFIG['model_name'],
                     pretrained=True,
                     target_size=CONFIG["num_classes"])
model.freeze()

print(CONFIG["lr"])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), CONFIG["lr"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


lr_finder = LRFinder(model, optimizer, loss_fn)
lr_finder.range_test(train_loader, end_lr=10, num_iter=200)
lr_finder.plot()
plt.show()


#trainer = Trainer(model,
#                  train_loader,
#                  valid_loader,
#                  loss_fn,
#                  device,
#                  Path(CONFIG["save_path"]),
#                  CONFIG["epoch_count"],
#                  CONFIG["lr"],
#                  optimizer)
#
#trainer.run()
