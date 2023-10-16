import sys
import torch


def remove_padding(lidar, threshold = 0.5):
    n_lidar = [] 

    for pc in lidar:
        #print("pc shape ", pc.shape)
        median_h = pc[:, 2].median()
        #print( median_h - (median_h*threshold))
        #print("median shape ", median_h.shape)
        idx = (pc[:, 2] >= median_h - (median_h*threshold)).nonzero(as_tuple=False)
        p = pc[idx].squeeze()
        #p = torch.gather(pc, dim = 0, index=idx)

        n_lidar.append(p)
    return n_lidar

