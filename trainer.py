import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from metric_monitor import MetricMonitor
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            loss_fn,
            device,
            save_path,
            epoch_count=1,
            lr=2e-3,
            optimizer=None,
            scheduler=None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epoch_count = epoch_count

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        else:
            self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_path = save_path


    def train_epoch(self):
        self.model.train()

        loss = MetricMonitor("train loss", 3)
        accuracy = MetricMonitor("train acc", 2)

        for image, label in tqdm(self.train_loader):
            image = image.to(self.device)
            target = label.to(self.device)

            output = self.model(image)
            acc = sum(torch.argmax(output, dim=1).eq(target)).cpu().numpy() / target.shape[0]

            self.optimizer.zero_grad()
            loss_value = self.loss_fn(output, target)

            loss_value.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            loss.update(loss_value.item())
            accuracy.update(acc * 100)

        return loss, accuracy

    def valid_epoch(self):
        self.model.eval()

        loss = MetricMonitor("valid loss", 3)
        accuracy = MetricMonitor("valid acc", 2)

        for image, label in tqdm(self.valid_loader):
            image = image.to(self.device)
            target = label.to(self.device)

            output = self.model(image)
            acc = sum(torch.argmax(output, dim=1).eq(target)).cpu().numpy() / target.shape[0]
            loss_value = self.loss_fn(output, target)

            loss.update(loss_value.item())
            accuracy.update(acc * 100)

        return loss, accuracy

    def run(self):

        for epoch_id in range(self.epoch_count):
            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.valid_epoch()
            print("[{}: {} | {} | {} | {}]".format(
                epoch_id,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc))

    def save_model_state(self):
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        model_path = self.save_path/"model.pth"
        torch.save(self.model.state_dict(), str(model_path))
