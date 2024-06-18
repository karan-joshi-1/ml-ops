import mlflow
import os, sys
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


class Template_Dataset(Dataset):
    def __init__(self, training_params):
        self.training_params = training_params
        self.df = self.loadLightfield(2, 2, "hogel")[0]  # For now, get the first df
        self.index = torch.from_numpy(
            self.df.index.values.astype(np.float32)
        )  # index values in the df (input)
        self.rgbTable = torch.from_numpy(  # rgb values (output)
            self.df[["r", "g", "b"]].values.astype(np.float32) / 255
        )

        self.len = self.index.shape[0]

    def __getitem__(self, index):
        return self.index[index], self.rgbTable[index]

    def __len__(self):
        return self.len

    # this function loads the lightField image file, and returns an array of
    # pandas dfs which each represent a subset of the image
    # cutby can be hogel, pixel, or angle
    # number of subRegions will automatically be caclulated to highest factor less than or equal to the specified
    # desired values
    def loadLightfield(
        self, desiredNumOfSubRegionsX, desiredNumOfSubRegionsY, cutby="hogel"
    ):
        # get the parameters from the JSON file
        dataset = str(self.training_params.get("modelParams").get("datasetPath"))
        rowHogel = int(
            self.training_params.get("modelParams").get("rowHogel")
        )  # number of rows of hogels
        columnHogel = int(
            self.training_params.get("modelParams").get("columnHogel")
        )  # number of cols of hogels

        image = cv.imread(dataset)

        self.width = image.shape[1]
        self.height = image.shape[0]

        self.resPerHogelY = int(self.height / rowHogel)
        self.resPerHogelX = int(self.width / columnHogel)

        shapedImage = image.reshape((-1, 3))

        df = pd.DataFrame(shapedImage)

        df["x"] = (df.index % self.width) // self.resPerHogelX
        df["y"] = (df.index // self.width) // self.resPerHogelY
        df["u"] = (df.index % self.width) % self.resPerHogelX
        df["v"] = (df.index // self.width) % self.resPerHogelY
        df["px"] = (
            df.index % self.width
        )  # TODO extra data, at some point this column should only be returned when debugging
        df["py"] = (
            df.index // self.width
        )  # TODO extra data, at some point this column should only be returned when debugging

        df = df.rename(columns={0: "r", 1: "g", 2: "b"})

        subDivisions = self.subDivideDataframe(
            df,
            desiredNumOfSubRegionsX,
            desiredNumOfSubRegionsY,
            cutby,
            columnHogel,
            rowHogel,
        )

        return subDivisions

    def subDivideDataframe(
        self,
        df,
        desiredNumOfSubRegionsX,
        desiredNumOfSubRegionsY,
        cutby,
        columnHogel,
        rowHogel,
    ):

        xRange = 0
        yRange = 0

        if cutby == "hogel":
            xRange = columnHogel
            yRange = rowHogel
            xCol = "x"
            yCol = "y"
        elif cutby == "angle":
            xRange = columnHogel
            yRange = rowHogel
            xCol = "u"
            yCol = "v"
        elif cutby == "pixel":
            xRange = self.width
            yRange = self.height
            xCol = "px"
            yCol = "py"

        subRegions = []

        numOfSubRegionsX = self.largestFactor(xRange, desiredNumOfSubRegionsX)
        numOfSubRegionsY = self.largestFactor(yRange, desiredNumOfSubRegionsY)

        xIncrement = math.ceil(xRange / numOfSubRegionsX)
        yIncrement = math.ceil(yRange / numOfSubRegionsY)
        for i in range(0, xRange, xIncrement):
            for j in range(0, yRange, yIncrement):
                subRegions.append(
                    df[
                        (df[xCol] >= i)
                        & (df[xCol] < (i + xIncrement))
                        & (df[yCol] >= j)
                        & (df[yCol] < (j + yIncrement))
                    ]
                )
        return subRegions

    # Given a number to factor and a desired maximum factor will return the closest factor that falls within that range
    def largestFactor(self, toFactor, maxFactor):
        if toFactor == maxFactor:
            return maxFactor

        divideFactor = 3
        if toFactor % 2 == 0:
            divideFactor = 2
        for i in range(toFactor // divideFactor, 0, -1):
            if toFactor % i == 0:
                if i <= maxFactor:
                    return i
