#!/usr/bin/env python3

import numpy as np
from termcolor import cprint

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1)[1:-1],
               np.linspace(p1[1], p2[1], parts+1)[1:-1])

def get_2d_grid(x_list, y_list, z_list):
    # create numpy 2d array of empty lists
    grid = np.empty((80, 80), dtype=object)
    for i in range(80):
        for j in range(80):
            grid[i][j] = []
    # populate the 2d array
    for x, y, z in zip(x_list, y_list, z_list):
        if abs(int(x)) >= 40 or abs(int(y)) >= 40: continue  # if out of bounds
        grid[int(x) + 40, -int(y) + 40].append(int(z))

    # empty lists get a zero
    for i in range(80):
        for j in range(80):
            if len(grid[i][j]) == 0:
                grid[i][j] = [0]

    return grid

class BEVLidar:
    def __init__(self, x_range=(-20, 20),
                 y_range=(-20, 20),
                 z_range=(-1, 5),
                 resolution=0.05,
                 threshold_z_range=False):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.dx = x_range[1]/resolution
        self.dy = y_range[1]/resolution
        self.img_size = int(1 + (x_range[1] - x_range[0]) / resolution)
        self.threshold_z_range = threshold_z_range
        cprint('created the bev image handler class', 'green', attrs=['bold'])

    def get_bev_lidar_img(self, lidar_points):
        img = np.zeros((self.img_size, self.img_size))
        for x, y, z in lidar_points:
            if self.not_in_range_check(x, y, z): continue
            ix = (self.dx + int(x / self.resolution))
            iy = (self.dy - int(y / self.resolution))
            if self.threshold_z_range:
                img[int(round(iy)), int(round(ix))] = 1 if z >= self.z_range[0] else 0
            else:
                img[int(round(iy)), int(round(ix))] = (z-self.z_range[0])/(self.z_range[1]-self.z_range[0])
        return img

    def not_in_range_check(self, x, y, z):
        if x < self.x_range[0] \
                or x > self.x_range[1] \
                or y < self.y_range[0] \
                or y > self.y_range[1] \
                or z < self.z_range[0] \
                or z > self.z_range[1]: return True
        return False

def get_bev_lidar(lidar_points, x_range=(-20, 20), y_range=(-20, 20), z_range=(-1, 5), resolution=0.05):
    """
    Function that returns a bev lidar image given lidar points and a range and resolution.
    """
    dx = x_range[1]/resolution
    dy = y_range[1]/resolution
    x_list, y_list, z_list = [], [], []

    for x, y, z in lidar_points:
        if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1] or z < z_range[0] or z > z_range[1]: continue
        xr = x/resolution
        yr = y/resolution

        ix = (dx + int(xr))
        iy = (dy - int(yr))

        x_list.append(round(ix))
        y_list.append(round(iy))
        z_list.append(z)

    # z_list = np.array(z_list)
    # z_list = (z_list - min(z_list.flatten())) / (max(z_list.flatten()) - min(z_list.flatten()))

    img_width = int(1 + (x_range[1] - x_range[0])/resolution)
    img_height = int(1 + (y_range[1] - y_range[0])/resolution)

    img = np.zeros((int(img_height), int(img_width)))

    # for x, y, z in zip(x_list, y_list, z_list):
    #     img[int(y), int(x)] = z
    #
    # # threshold image
    # img[img < 0.01] = 0
    # img[img >= 0.01] = 1

    for x, y, z in zip(x_list, y_list, z_list):
        img[int(y), int(x)] = 1 if z >= 0.01 else 0

    return img


def get_height_variance_map(x_list, y_list, z_list):
    height_variance = get_2d_grid(x_list, y_list, z_list)
    for i in range(80):
        for j in range(80):
            height_variance[i][j] = np.std(height_variance[i][j])
    height_variance = height_variance / max(height_variance.flatten())
    return (height_variance.T).astype(np.float32)

def get_height_mean_map(x_list, y_list, z_list):
    height_mean = get_2d_grid(x_list, y_list, z_list)
    for i in range(80):
        for j in range(80):
            height_mean[i][j] = np.mean(height_mean[i][j])
    height_mean = height_mean / max(height_mean.flatten())
    return (height_mean.T).astype(np.float32)

def get_cell_occupancy_map(x_list, y_list, z_list):
    # create numpy 2d array of empty lists
    grid = np.empty((80, 80), dtype=object)
    for i in range(80):
        for j in range(80):
            grid[i][j] = 0

    # populate the 2d array
    for x, y, z in zip(x_list, y_list, z_list):
        if abs(int(x)) >= 40 or abs(int(y)) >= 40: continue  # if out of bounds
        grid[int(x) + 40, -int(y) + 40] = 255.0

        eqpts = getEquidistantPoints([0, 0], [x, y], 50)
        for x1, y1 in eqpts:
            grid[int(x1) + 40, -int(y1) + 40] += 255.0
            grid[int(x1) + 40, -int(y1) + 40] = min(255.0, grid[int(x1) + 40, -int(y1) + 40])

    grid = grid / max(grid.flatten())
    return (grid.T).astype(np.float32)


def get_start_end_times_from_bag(bag, topic='/velodyne_points'):
    """
    Get the start and end time of the bag file.
    """
    start_time, end_time = None, None
    for topic, msg, t in bag.read_messages(topics=[topic]):
        if start_time is None: start_time = t
        end_time = t
    return start_time, end_time
