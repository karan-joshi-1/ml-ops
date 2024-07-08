import csv
import mlflow
import os
import sys
import json
import datetime
import time
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np

import cv2 as cv
import pandas as pd
import math

from json_helper import log_print, read_json
from dataset import Template_Dataset


class __Model__(nn.Module):
    def __init__(self, dataset):
        super(__Model__, self).__init__()

        self.xyLayer = nn.Sequential(
            nn.Linear(2, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.uvLayer = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(192 + 256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 3),
            nn.Sigmoid(),
        )

        self.pixelPerHogelX = dataset.width / dataset.resPerHogelX
        self.pixelPerHogelY = dataset.height / dataset.resPerHogelY
        self.hResX = dataset.resPerHogelX
        self.hResY = dataset.resPerHogelY
        self.len = dataset.index.shape[0]
        self.width = dataset.width

    def forward(self, index):
        col = index % self.width
        row = torch.div(index, self.width, rounding_mode="floor")

        hx = torch.div(col, self.pixelPerHogelX, rounding_mode="floor")
        hy = torch.div(row, self.pixelPerHogelY, rounding_mode="floor")

        hx = torch.div(hx, self.hResX)
        hy = torch.div(hy, self.hResY)

        u = torch.div(col % self.pixelPerHogelX, self.pixelPerHogelX)
        v = torch.div(row % self.pixelPerHogelY, self.pixelPerHogelY)

        xy = torch.cat((hx, hy), 0)
        uv = torch.cat((u, v), 0)

        xy = xy.view(2, -1)
        xy = xy.transpose(0, 1)

        uv = uv.view(2, -1)
        uv = uv.transpose(0, 1)
        xy = self.xyLayer(xy)
        uv = self.uvLayer(uv)
        index = self.fc(torch.cat((xy, uv), 1))

        return index


class Trainer:
    def __init__(self, training_params, logToMlFlow=True, csv_log_path='training_log.csv'):
        self.training_params = training_params
        self.run_name = None
        self.logToMlFlow = logToMlFlow
        self.csv_log_path = csv_log_path

        self.dataset = Template_Dataset(self.training_params)
        self.createModel()
        self.initialize_csv_logging()

    def createModel(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_print("Using", torch.cuda.device_count(), "GPUs!")

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(__Model__(self.dataset)).to(self.device)
            log_print("multiple gpus")
        else:
            self.model = __Model__(self.dataset).to(self.device)
            log_print("single gpu")

    def initialize_csv_logging(self):
        with open(self.csv_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = ["epoch", "training_time", "training_loss"]
            writer.writerow(headers)

    def log_to_csv(self, epoch_index, epoch_start, epoch_end, epoch_loss):
        with open(self.csv_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            training_time = epoch_end - epoch_start
            writer.writerow([epoch_index, training_time, epoch_loss])

    def initializeTraining(self):
        self.logInitialize()

    def train(self):
        criterion = torch.nn.MSELoss()
        optimizer = Adam(
            self.model.parameters(),
            lr=self.training_params.get("modelParams").get("lr"),
            betas=self.training_params.get("modelParams").get("betas"),
        )

        dataLoader = DataLoader(
            dataset=self.dataset,
            batch_size=self.training_params.get("modelParams").get("batch_size"),
            num_workers=2,
            pin_memory=True,
        )

        epochs = int(self.training_params.get("modelParams").get("epochs"))

        for epoch_index in range(epochs):
            try:
                self.trainEpoch(epoch_index, dataLoader, criterion, optimizer, epochs)
            except Exception as e:
                log_print(f"Exception {e}")

    def trainEpoch(self, epoch_index, dataLoader, criterion, optimizer, epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for index, rgb_values in dataLoader:
            index = index.to(self.device)
            rgb_values = rgb_values.to(self.device)

            y_pred = self.model(index)
            loss = criterion(y_pred, rgb_values)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end = time.time()
        epochMetrics = {
            f"Epoch Training Time": epoch_end - epoch_start,
            f"Epoch Training Loss": epoch_loss,
        }

        self.logOnEpoch(epoch_index, epochMetrics)
        self.logOnEpochInterval(epoch_index, epoch_loss, epochs)
        self.log_to_csv(epoch_index, epoch_start, epoch_end, epoch_loss)

    def log_final_metrics(self):
        final_metrics = {
            "Final Training Loss": self.training_params.get("final_training_loss", 0.0),
            "Final Validation Accuracy": self.training_params.get("final_validation_accuracy", 0.0),
        }
        if self.logToMlFlow:
            for key in final_metrics:
                mlflow.log_metric(key, final_metrics[key])

        with open(self.csv_log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for key, value in final_metrics.items():
                writer.writerow([key, value])

    def endTraining(self):
        self.log_final_metrics()
        self.logEnd()

    def logInitialize(self):
        if self.logToMlFlow:
            branchName = str(self.training_params.get("gitParams").get("branchName"))
            dateTimeString = str(datetime.datetime.now().strftime("%m_%d_%y-%H_%M"))
            self.run_name = branchName + "-" + dateTimeString
            mlflow.set_tracking_uri("http://10.220.115.62:5000/")
            mlflow.start_run(run_name=self.run_name)
            mlflow.log_params(self.training_params)

    def logOnEpoch(self, epoch_index, metrics):
        if self.logToMlFlow:
            for key in metrics:
                mlflow.log_metric(key, metrics[key], epoch_index)

    def logOnEpochInterval(self, epoch_index, epoch_loss, epochs):
        if self.logToMlFlow:
            log_interval = int(
                self.training_params.get("modelParams").get("logInterval")
            )

            if (
                epoch_index % log_interval == 0 and epoch_index != 0
            ) or epoch_index == (epochs - 1):
                branchName = str(
                    self.training_params.get("gitParams").get("branchName")
                )

                checkpointPath = "ModelCheckpoints/" + branchName
                if not os.path.exists(checkpointPath):
                    os.makedirs(checkpointPath)

                checkpoint_path = os.path.join(
                    checkpointPath,
                    (
                        "epoch_"
                        + str(epoch_index)
                        + "_date_"
                        + datetime.datetime.now().strftime("%m_%d_%y-%H_%M")
                        + ".pt"
                    ),
                )

                checkpointParams = {
                    f"Checkpoint Path for Epoch {epoch_index}": checkpoint_path
                }
                for key in checkpointParams:
                    mlflow.log_param(key, checkpointParams[key])

                state_dict = None

                if isinstance(self.model, nn.DataParallel):
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                torch.save(
                    {
                        "epoch": epoch_index,
                        "model_state_dict": state_dict,
                        "loss": epoch_loss,
                    },
                    checkpoint_path,
                )

    def logEnd(self):
        if self.logToMlFlow:
            mlflow.end_run()


if __name__ == "__main__":
    try:
        log_print(datetime.datetime.now().strftime("%m_%d_%y-%H_%M"))
        file = open("modelConfig.json")
        training_params = json.load(file)
        file.close()

        trainer = Trainer(training_params)
        trainer.initializeTraining()
        trainer.train()
        trainer.endTraining()

    except Exception as err:
       
