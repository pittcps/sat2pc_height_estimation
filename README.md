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

# Usage

To train the model you can run the following command:

```
python train.py --data-dir ./sat2pc_height_estiamtion_dataset/
```

To test the model you can run the following command:

```
python test.py --data-dir ./sat2pc_height_estiamtion_dataset/ --ckpt-path [location of the saved weights]
```