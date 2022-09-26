"""
UNIT中GAN的复现
"""

from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import torch


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 24)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

    def cal_dis_loss(self, input_fake, input_real):
        # calculate the loss to train discriminator D
        out_set0 = self.forward(input_fake)
        out_set1 = self.forward(input_real)
        loss = 0
        for (out0, out1) in zip(out_set0, out_set1):
            loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)  # LSGAN
        return loss

    def cal_gen_loss(self, input_fake):
        # calculate the loss to train generator G
        out_set0 = self.forward(input_fake)
        loss = 0
        for out0 in out_set0:
            loss += torch.mean((out0 - 1) ** 2)
        return loss


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        hidden = self.encoder(x)
        if self.training:
            noise = Variable(torch.randn(hidden.size()))
            x_recon = self.decoder(hidden + noise)
        else:
            x_recon = self.decoder(hidden)
        return x_recon, hidden

    def encode(self, x):
        hidden = self.encoder(x)
        noise = Variable(torch.randn(hidden.size()))
        return hidden, noise

    def decode(self, hidden):
        x_recon = self.decoder(hidden)
        return x_recon


# Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 24)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# Dataset
class TrainSet(Dataset):
    def __init__(self, list_x_a, list_x_b):
        super(TrainSet, self).__init__()
        self.list_x_a = list_x_a
        self.list_x_b = list_x_b

    def __getitem__(self, index):
        return self.list_x_a[index], self.list_x_b[index]

    def __len__(self):
        return len(self.list_x_b)

