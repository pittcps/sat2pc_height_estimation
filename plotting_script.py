import matplotlib.pyplot as plt
import numpy as np
import json
import data_util
import pptk
import utility
import torch
import os


def plot_single_dist(dict, x_label, y_label, title, legends):
    plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
    plt.rcParams.update({'font.size': 40})
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
    plt.rcParams['font.family'] = 'sans-serif' # ... for regular text


    fig, ax = plt.subplots()
    for i, key in enumerate(legends):
        dict[key].sort()
        bins = list(range(0, 1001, 10))
        hist, bin_edges = np.histogram(dict[key], bins=bins)
        hist = [0] + list(hist)

        plt.plot(bin_edges, hist, linewidth=10.0, label=key)

    
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_box_chart(data, x_label, y_label, title, x_ticks):
    fig = plt.figure(figsize =(10, 7))
 
    # Creating plot
    plt.boxplot(data)
    
    # show plot
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.xticks([i+1 for i in range(len(x_ticks))], x_ticks)

    plt.show()

def plot_height_dist(height_data_dir):
    heights_dict = data_util.load_json(height_data_dir)

    means = [heights_dict[key]['mean'] for key in heights_dict.keys()]
    print(len(means))

    plot_single_dist({'Height Means': means}, 'Mean Height', 'Number of Buildings', '', ['Height Means'])

    tall = short = very_short = 0
    for m in means:
        if m > 450:
            tall += 1
        elif m > 300:
            short += 1
        else:
            very_short += 1
    print("tall ", tall)
    print("short ", short)
    print("very_short ", very_short)

def run_qualitative_loss(imgs, gt_point_clouds, sat2pc_point_clouds):

    for i in range(len(gt_point_clouds)):
        gt_point_cloud = gt_point_clouds[i]
        sat2pc_point_cloud = sat2pc_point_clouds[i]
        img = imgs[i]

        scale = [min(gt_point_cloud[:, 2]), max(gt_point_cloud[:, 2])]

        img.show()

        v1 = pptk.viewer(gt_point_cloud, gt_point_cloud[:, 2])
        v1.color_map('jet', scale = scale)
        v1.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

        v2 = pptk.viewer(sat2pc_point_cloud, sat2pc_point_cloud[:, 2])
        v2.color_map('jet', scale = scale)
        v2.set(point_size=10, bg_color=[0, 0, 0, 0], show_grid = False, show_info = False, show_axis= False)

        input("Please press enter to visualize the next sample ...")
        img.close()
        v1.close()
        v2.close()