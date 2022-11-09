import numpy as np


def get_dist_mat(coord: list):
    dist_mat = {}
    for i in range(len(coord)):
        for j in range(len(coord)):
            dist_mat[(i, j)] = np.round(((coord[i][0] - coord[j][0]) ** 2 + (coord[i][1] - coord[j][1]) ** 2) ** 0.5, 2)
    return dist_mat
