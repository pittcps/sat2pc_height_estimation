import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Sat2HeightMultiFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_ftrs // 2)

        self.fc1 = nn.Linear(num_ftrs // 2, 2)
        self.th = nn.Tanh()

    def forward(self, x):
        x = self.resnet(x)
        x = self.th(x)
        x = self.fc1(x)
        return x


class Sat2HeightTwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc_mean = nn.Linear(num_ftrs, 1)
        self.fc_std = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet(x)
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        out = torch.cat([mean, std], 1)
        return out


class Sat2HeightFourBranch(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        print("Using backbone", model_name)
        if model_name == "vgg16":
            self.resnet = models.vgg16(pretrained=True)
            self.resnet.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

            modules = []
            modules.append(nn.Linear(25088, 4096, bias=True))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Dropout(p=0.5, inplace=False))
            sequential = nn.Sequential(*modules)
            self.resnet.classifier = sequential
            num_ftrs = 4096
        elif model_name == "vgg19":
            self.resnet = models.vgg19(pretrained=True)
            self.resnet.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

            modules = []
            modules.append(nn.Linear(25088, 4096, bias=True))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Dropout(p=0.5, inplace=False))
            sequential = nn.Sequential(*modules)
            self.resnet.classifier = sequential
            num_ftrs = 4096

        elif model_name == "resnet101":
            self.resnet = models.resnet101(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
        else:
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
        print(self.resnet)

        self.fc_z_mean = nn.Linear(num_ftrs, 1)
        self.fc_z_std = nn.Linear(num_ftrs, 1)
        self.fc_x_std = nn.Linear(num_ftrs, 1)
        self.fc_y_std = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x = self.resnet(x)
        z_mean = self.fc_z_mean(x)
        z_std = self.fc_z_std(x)
        x_std = self.fc_x_std(x)
        y_std = self.fc_y_std(x)
        out = torch.cat([z_mean, z_std, x_std, y_std], 1)
        return out


class Sat2HeightSingleFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x


def get_model(four_branch, model_name=None):
    if four_branch:
        model = Sat2HeightFourBranch(model_name)
    else:
        model = Sat2HeightTwoBranch()

    return model