"""
Training the Discirminator and Generator from DCGAN paper
"""

import torch
from torch import nn
from torch.optim import optim

import torchvision
from torchvision import datasets , transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator , Discriminator , initialise_wt

# DEVCE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

#HYPERPARAMETERS
LR = 2e-4
BATCH_SIZE = 128
IMG_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for i in range(CHANNELS_IMG)] , [0.5 for i in range(CHANNELS_IMG)]
        )
    ]
)

