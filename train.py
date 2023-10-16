import torch
from torchvision import datasets, models, transforms
import copy
import torch.nn as nn
import torch.optim as optim
import resnet_model
from torch.optim import lr_scheduler
import glob
import re
import os
from sat2height_dataset import Sat2HeightDataset

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default=".\\sat2pc_height_estiamtion_dataset")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_ckpt(model, optimizer, lr_scheduler, epochs, ckpt_path, **kwargs):
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"]  = optimizer.state_dict()
    checkpoint["epochs"] = epochs
    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        
    for k, v in kwargs.items():
        checkpoint[k] = v
        
    prefix, ext = os.path.splitext(ckpt_path)
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)

    torch.save(checkpoint, ckpt_path)

    # it will create many checkpoint files during training, so delete some.
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    n = 2
    if len(ckpts) > n:
        for i in range(len(ckpts) - n):
            os.remove(ckpts[i])
            
    if 'best' in kwargs and kwargs['best']==True:
        prefix = prefix+"_best"    
    ckpt_path = "{}-{}{}".format(prefix, epochs, ext)

    torch.save(checkpoint, ckpt_path)

    # it will create many checkpoint files during training, so delete some.
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    n = 2
    if len(ckpts) > n:
        for i in range(len(ckpts) - n):
            os.remove(ckpts[i])
    


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, ckpt_path, num_epochs=25, four_branch = True):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999999.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, (imgs, masks, labels) in enumerate(dataloaders[phase]):
                imgs = imgs.to(device)
                masks = masks.to(device)
                masks = masks.unsqueeze(1)
                inputs = torch.cat((imgs, masks), dim=1)
                
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss_z_means = criterion(outputs[:, 0], labels[:, 0])
                    loss_z_stds = criterion(outputs[:, 1], labels[:, 1])
                    if four_branch:
                        loss_x_stds = criterion(outputs[:, 2], labels[:, 2])
                        loss_y_stds = criterion(outputs[:, 3], labels[:, 3])
                        loss = loss_z_means + loss_z_stds + loss_x_stds + loss_y_stds
                    else:
                        loss = loss_z_means + loss_z_stds
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss and epoch > 9:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_ckpt(model, optimizer, scheduler, epoch, ckpt_path, best=True)
            elif phase == 'val':
                save_ckpt(model, optimizer, scheduler, epoch, ckpt_path)


    print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run_train():

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    model = resnet_model.get_model(four_branch = True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    num_epochs = 100
    batch_size = 4

    dataloaders = {}
    _d_train = Sat2HeightDataset(
        image_dir = os.path.join(args.data_dir, os.path.join('train', 'image')), 
        ann_dir = os.path.join(args.data_dir, os.path.join('train', 'annotation')), 
        label_dir = os.path.join(args.data_dir, 'roof_height_mean_and_std_centimeters.json'),
        mode = 'train',
        bootstarp = True,
        transform = data_transforms['train'])
    _d_val = Sat2HeightDataset(
        image_dir = os.path.join(args.data_dir, os.path.join('val', 'image')),
        ann_dir = os.path.join(args.data_dir, os.path.join('val', 'annotation')),
        label_dir = os.path.join(args.data_dir, 'roof_height_mean_and_std_centimeters.json'),
        mode = 'val',
        bootstarp = False,
        transform = data_transforms['val'])
    
    print("Number of training samples ", len(_d_train))
    print("Number of valdiation samples ", len(_d_val))
    
    dataset_sizes = {'train': _d_train.get_number_of_samples(), 'val': _d_val.get_number_of_samples()}

    dataloaders['train'] = torch.utils.data.DataLoader(_d_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloaders['val'] = torch.utils.data.DataLoader(_d_val, batch_size=1, shuffle=True, num_workers=0)

    ckpt_path = './model_weights.chpt'

    train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, ckpt_path, num_epochs)


run_train()

