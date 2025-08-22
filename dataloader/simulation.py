import numpy as np
import pandas as pd
import math
import pickle
import json
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset


def load_data():
    '''
    Load simulation numpy files
    '''
    polair = np.load('data/d_polair.npy')
    polair_o3 = np.load('data/d_polair_O3.npy')
    polair_pm10 = np.load('data/d_polair_PM10.npy')
    polair_pm25 = np.load('data/d_polair_PM25.npy')

    all_polair = np.concatenate((polair, polair_o3, polair_pm10, polair_pm25), axis=1)

    return all_polair


def sensor_location():
    '''
    Outputs the list of sensor location (x,y) on grid 
    '''
    sensor_pos = []
    channels = ['NO2', 'O3', 'PM10', 'PM25']
    for c in channels:
        with open("data/new_observation_data/station_data_"+c+ ".pkl", "rb") as f:
            station_data_dict = pickle.load(f)
        coords = []

        for station in station_data_dict.values():
            coords.append([station['x_index'], station['y_index']])
        
        sensor_pos.append(np.array(coords))

    return sensor_pos

def data_stat():
    '''
    Loads the population data statistics of simulation data
    '''
    with open('utils/minmax.json', 'r') as file:
        dict_minmax = json.load(file)
    dict_minmax = {int(k): v for k, v in dict_minmax.items()}

    with open('utils/mean_std.json', 'r') as file:
        dict_meanstd = json.load(file)
    dict_meanstd = {int(k): v for k, v in dict_meanstd.items()}

    return dict_minmax, dict_meanstd

def create_dataset(normalisation, train_idx):
    '''
    normalisation: True if gaussian norm applied, False if min-max norm applied

    returns: Torch Dataset
    '''
    all_polair = load_data()
    sensor_pos = sensor_location()
    dict_minmax, dict_meanstd = data_stat()

    if normalisation:
        train_dataset_polair = AirPollutionDataset(all_polair[:train_idx], mean_std_data=dict_meanstd, dict_min_max=None, sensor_pos=sensor_pos, vt=True, random_mask=False)
        val_dataset_polair = AirPollutionDataset(all_polair[train_idx:], mean_std_data=dict_meanstd, dict_min_max=None, sensor_pos=sensor_pos, vt=True, random_mask=False)
    else:
        train_dataset_polair = AirPollutionDataset(all_polair[:train_idx], mean_std_data=None, dict_min_max=dict_minmax, sensor_pos=sensor_pos, vt=True, random_mask=False)
        val_dataset_polair = AirPollutionDataset(all_polair[train_idx:], mean_std_data=None, dict_min_max=dict_minmax, sensor_pos=sensor_pos, vt=True, random_mask=False)
    
    return train_dataset_polair, val_dataset_polair
#
#

class AirPollutionDataset(Dataset):
    def __init__(self, data, mean_std_data, dict_min_max, sensor_pos=None, random_mask=False, mask_ratio=0.9, vt=True):
        
        self.data = data
        self.sensor_pos = sensor_pos
        self.mean_std_data = mean_std_data
        self.dict_min_max = dict_min_max
        self.random_mask = random_mask
        self.mask_ratio = mask_ratio
        self.vt = vt

    def __len__(self):
        return self.data.shape[0]
    
    def normalize(self, img):
        C, H, W = img.shape
        for i in range(C):
            img[i] = 0.5*(img[i]-self.mean_std_data[i][0])/self.mean_std_data[i][1]

        return img
    
    def standardize(self, img):
        C, H, W = img.shape
        for i in range(C):
            img[i] = (img[i]-self.dict_min_max[i][0])/(self.dict_min_max[i][1] - self.dict_min_max[i][0])

        return img
    
    def voronoi_tess(self, img, mask):
        C, H, W = img.shape  #8518, 1, 75, 110
        grid_x, grid_y = np.mgrid[0:H, 0:W]

        new_img = np.zeros_like(img)

        for channel in range(C):
            flatten_channel_idx, flatten_y_idx, flatten_x_idx = torch.nonzero(mask, as_tuple=True)
            target_idx = flatten_channel_idx == channel
            points = torch.column_stack((flatten_x_idx[target_idx], flatten_y_idx[target_idx]))
        
            values = img[channel]

            x = points[:,1]
            y = points[:,0]
            values = values[x,y]
            new_img[channel] = griddata(points[:,[1,0]], values, (grid_x, grid_y), method='nearest')
            

        return (torch.from_numpy(new_img)).to(torch.float32)
    
    def masking(self, image, index):
        np.random.seed(index)
        C, H, W = image.shape  # channel, length, dim
        if self.sensor_pos:
                mask = torch.zeros_like(image).to(image.device)
                for i in range(C):
                    arr = self.sensor_pos[i]
                    keep_points = math.floor(arr.shape[0]*0.85)

                    must_keep_mask = ((arr[:, 0] < 20) | (arr[:, 0] > 80)) & (arr[:, 1] > 55)
                    must_keep = arr[must_keep_mask]
                    remaining = arr[~must_keep_mask]

                    num_to_sample = keep_points - len(must_keep)
                    selected_additional = remaining[np.random.choice(len(remaining), size=num_to_sample, replace=False)]
                    final_selected = np.concatenate([must_keep, selected_additional], axis=0)

                    for x,y in final_selected:
                            mask[i,y,x] = 1
                
                masked_img = image*mask
                return masked_img, mask
            
        if self.random_mask:
                mask = (torch.rand(1, H, W) > self.mask_ratio).to(image.device)  # Retain (1-mask_ratio)% of pixels
        else:
                torch.manual_seed(42)
                mask = (torch.rand(1, H, W) > self.mask_ratio).to(image.device)
        
        masked_img = image * mask

        return masked_img, mask

    
    def __getitem__(self, index):
        img = self.data[index].copy()

        if self.mean_std_data:
            img = self.normalize(img)
        else:
             img = self.standardize(img)

        img = (torch.from_numpy(img)).to(torch.float32)
        masked_img, mask = self.masking(img, index)

        if self.vt:
            vt = self.voronoi_tess(img, mask)
        else:
            vt = None

        return img, mask, vt
