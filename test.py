import torch
from torchvision import datasets, models, transforms
import copy
import torch.nn as nn
import torch.optim as optim
import resnet_model
from torch.optim import lr_scheduler
import os
from sat2height_dataset import Sat2HeightDataset
import data_util
from PIL import Image
import numpy as np
import argparse
import plotting_script
import json
import utility

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default="./dataset")
parser.add_argument("--backbone", default="resnet50")
# parser.add_argument("--data-dir", default=".\\neighbourhood_test")
parser.add_argument("--ckpt-path", default="./checkpoint/model_weights-resnet101.chpt")
parser.add_argument("--result", default="./results/test.json")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    _d_test = Sat2HeightDataset(
        image_dir=os.path.join(args.data_dir, os.path.join('test', 'image')),
        ann_dir=os.path.join(args.data_dir, os.path.join('test', 'annotation')),
        label_dir=os.path.join(args.data_dir, 'roof_height_mean_and_std_centimeters.json'),
        mode='test',
        bootstarp=False,
        transform=data_transform
    )
    dataset_size = _d_test.get_number_of_samples()
    dataloaders = torch.utils.data.DataLoader(_d_test, batch_size=1, shuffle=True, num_workers=0)

    model = resnet_model.get_model(True, args.backbone).to(device)
    # model = resnet_model.Sat2Height().to(device)

    if args.ckpt_path:
        checkpoint = torch.load(args.ckpt_path, map_location=device)  # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print('No checkpoints')

    criterion = nn.MSELoss()
    MAE = nn.L1Loss()

    running_MSE_loss_on_z_mean = 0.0
    running_MSE_loss_on_z_std = 0.0
    running_MSE_loss_on_x_std = 0.0
    running_MSE_loss_on_y_std = 0.0
    running_MAE_loss_on_z_mean = 0.0
    running_MAE_loss_on_z_std = 0.0
    running_MAE_loss_on_x_std = 0.0
    running_MAE_loss_on_y_std = 0.0

    max_z_mean_error = 0
    max_z_std_error = 0

    all_height_labels = []
    all_height_predictions = []
    all_height_error = []
    all_x_std_predictions = []
    all_y_std_predictions = []
    tall_errors = []
    short_errors = []
    sample_to_result = {}

    for i, (imgs, masks, labels, img_ids) in enumerate(dataloaders):
        imgs = imgs.to(device)
        masks = masks.to(device)
        masks = masks.unsqueeze(1)
        inputs = torch.cat((imgs, masks), dim=1)
        labels = labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)

            z_means = outputs[:, 0]
            z_means_label = labels[:, 0]
            z_stds = outputs[:, 1]
            z_stds_label = labels[:, 1]
            x_stds = outputs[:, 2]
            x_stds_label = labels[:, 2]
            y_stds = outputs[:, 3]
            y_stds_label = labels[:, 3]

            all_height_labels.append(z_means_label.item())
            all_height_predictions.append(z_means.item())
            all_x_std_predictions.append(x_stds.item())
            all_y_std_predictions.append(y_stds.item())
            # all_height_error.append(means_label.item() - means.item())
            all_height_error.append(abs(z_means_label.item() - z_means.item()))

            MSE_loss_on_z_mean = criterion(z_means, z_means_label)
            MSE_loss_on_z_std = criterion(z_stds, z_stds_label)
            MSE_loss_on_x_std = criterion(x_stds, x_stds_label)
            MSE_loss_on_y_std = criterion(y_stds, y_stds_label)

            running_MSE_loss_on_z_mean += MSE_loss_on_z_mean.item() * inputs.size(0)
            running_MSE_loss_on_z_std += MSE_loss_on_z_std.item() * inputs.size(0)
            running_MSE_loss_on_x_std += MSE_loss_on_x_std.item() * inputs.size(0)
            running_MSE_loss_on_y_std += MSE_loss_on_y_std.item() * inputs.size(0)

            MAE_loss_on_z_mean = MAE(z_means, z_means_label)
            MAE_loss_on_z_std = MAE(z_stds, z_stds_label)
            MAE_loss_on_x_std = MAE(x_stds, x_stds_label)
            MAE_loss_on_y_std = MAE(y_stds, y_stds_label)

            if MAE_loss_on_z_mean.item() > max_z_mean_error:
                max_z_mean_error = MAE_loss_on_z_mean.item()

            if MAE_loss_on_z_std.item() > max_z_std_error:
                max_z_std_error = MAE_loss_on_z_std.item()

            if z_means_label.item() > 200:
                tall_errors.append(MAE_loss_on_z_mean.item())
            else:
                short_errors.append(MAE_loss_on_z_mean.item())

            running_MAE_loss_on_z_mean += MAE_loss_on_z_mean.item() * inputs.size(0)
            running_MAE_loss_on_z_std += MAE_loss_on_z_std.item() * inputs.size(0)
            running_MAE_loss_on_x_std += MAE_loss_on_x_std.item() * inputs.size(0)
            running_MAE_loss_on_y_std += MAE_loss_on_y_std.item() * inputs.size(0)

            sample_to_result[img_ids[0].item()] = {'z_mean': z_means.item(), 'z_std': z_stds.item(),
                                                   'x_std': x_stds.item(), 'y_std': y_stds.item()}

            # print("---------------------------------")
            # print("Label: ", labels)
            # print("Output: ", outputs)

    print(min(all_height_predictions), max(all_height_predictions))
    print(min(all_height_labels), max(all_height_labels))
    import statistics
    var = statistics.variance(all_height_error)
    print("Numeber of tall buildings: ", len(tall_errors))
    plotting_script.plot_single_dist({'Ground Truth Height Distribution': all_height_labels,
                                      'Prediction Height Distribution': all_height_predictions},
                                     'Height (cm)', 'Number of Buildings', 'Prediction and Label Distribution',
                                     ['Ground Truth Height Distribution', 'Prediction Height Distribution'])
    plotting_script.plot_box_chart([all_height_error, short_errors, tall_errors], 'Absolute Error', 'Value(cm)',
                                   'Absolute Error Distribution',
                                   ['All Buildigns', 'Short Buildings', "Tall Buildings"])

    test_MSE_on_z_mean = running_MSE_loss_on_z_mean / dataset_size
    test_MSE_on_z_std = running_MSE_loss_on_z_std / dataset_size
    test_MSE_on_x_std = running_MSE_loss_on_x_std / dataset_size
    test_MSE_on_y_std = running_MSE_loss_on_y_std / dataset_size

    test_MAE_on_z_mean = running_MAE_loss_on_z_mean / dataset_size
    test_MAE_on_z_std = running_MAE_loss_on_z_std / dataset_size
    test_MAE_on_x_std = running_MAE_loss_on_x_std / dataset_size
    test_MAE_on_y_std = running_MAE_loss_on_y_std / dataset_size

    print(f'Test MSE on mean heights: {test_MSE_on_z_mean:.4f}, on std: {test_MSE_on_z_std:.4f}')
    print(f'Test MSE on x std: {test_MSE_on_x_std:.4f}, on y std: {test_MSE_on_y_std:.4f}')
    print(f'Test MAE on mean heights: {test_MAE_on_z_mean:.4f}, on std: {test_MAE_on_z_std:.4f}')
    print(f'Test MAE on x std: {test_MAE_on_x_std:.4f}, on y std: {test_MAE_on_y_std:.4f}')
    print(f'Max mean height error: {max_z_mean_error:.4f}, max std error: {max_z_std_error:.4f}')

    with open(args.result, "w") as outfile:
        json.dump(sample_to_result, outfile)


test()