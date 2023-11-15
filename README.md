# Sat2Map

By Yoones Rezaei, Stephen Lee



# Introduction

In this repository we release the code and data for our paper "sat2Map: Reconstructing 3D Building Roof from 2D Satellite Images" which is an extension to [sat2pc](https://github.com/pittcps/sat2pc).

# Installation

To use this repository you need an environemnt with python 3.7.9. We suggests creating a conda environment with the following command:

```
conda create -n "sat2Map" python=3.7.9
```

Next, activate the environemnt and cd to the direcotry of the downloaded repository then run the following command:

```
python create_conda_environemt.py
```

This command will install the required packages.

# Data

The dataset from the paper can be dowloaded from [here](https://drive.google.com/file/d/1Kx5-Z3UZpd78nau1XAh5vK5TeBH4yB8k/view?usp=sharing).

# Model

The models for the paper can be downloaded using the link below. 
- VGG-16 model [here](https://pitt-my.sharepoint.com/:u:/g/personal/stl86_pitt_edu/ERqh60I5F7FEh8wKlULSh2UBNcEPo0tSKW5wzxppdNk7EQ?e=2A2UyB)
- VGG-19 model [here](https://pitt-my.sharepoint.com/:u:/g/personal/stl86_pitt_edu/EaF8aUM06OhGoxaI7JnpZ8gB4GCBXtBPPZGOY9KjfxJ4Ww)
- Resnet-50 model [here](https://pitt-my.sharepoint.com/:u:/g/personal/stl86_pitt_edu/EZDepM2EoOFBgwPFSeSTQjUB7PaARv1cGO3cLZ4c05x6BA?e=blQ4rY) 
- Resnet-101 model [here](https://pitt-my.sharepoint.com/:u:/g/personal/stl86_pitt_edu/EcbL52MrMOpOj-_ObG1TtS4B6HWOsrJG8Iyn5n1u33ZWXQ?e=6uzPgj)

# Usage

To train the model you can run the following command:

```
python train.py --data-dir ./sat2pc_height_estiamtion_dataset/
```

To test the model you can run the following command:

```
python test.py --data-dir ./sat2pc_height_estiamtion_dataset/ --ckpt-path [location of the saved weights]
```