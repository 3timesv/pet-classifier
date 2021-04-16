import os
import sys
from pathlib import Path

import yaml

import torch
sys.path.append(os.getcwd())
import utils
from dataset import Loader
from models import CustomResnet
from predictor import Predictor
from preprocess import PetTransform


with open("config.yaml", 'r') as config_file:
    CONFIG = yaml.safe_load(config_file)


save_predictions_path = Path("saved/predictions")

predict_path = Path("data/images")

#path_list = list(predict_path.glob("*.jpg"))[:10]
#print(path_list)

path = Path("data/images/english_setter_62.jpg")
path_list = [path]


model = CustomResnet(CONFIG['model_name'],
                     pretrained=False,
                     target_size=CONFIG["num_classes"])

data_loader = Loader(path_list,
                     label_func=utils.get_label,
                     batch_size=CONFIG["batch_size"],
                     shuffle=False,
                     transform=PetTransform.valid)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

predictor = Predictor(model,
                      data_loader,
                      device,
                      save_predictions_path)

predictor.run("saved/model_state/baseline_model.pth")

