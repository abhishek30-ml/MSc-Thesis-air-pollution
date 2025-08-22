import numpy as np
import pandas as pd
import math
import pickle
import json
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset


def create_df_list():
    '''
    Load observation data pickle file
    Merge the dataframe date-wise 
    Outputs a list of dataframe containing observation value for each monitoring station for each pollutant
    '''
    channels = ['NO2', 'O3', 'PM10', 'PM25']
    all_df_list = []
    for c in channels:
        with open("data/new_observation_data/station_data_"+ c + ".pkl", "rb") as f:
            station_data_dict = pickle.load(f)

        df_list = []
        for i, station in enumerate(station_data_dict.values()):
            df = station['obs']
            df = df.rename(columns={'valeur':'valuer'+str(i)})
            df_list.append(df)

        df_final = df_list[0]
        for i in range(1,len(df_list)):
            df_final = pd.merge(df_final, df_list[i], on='Date de début', how='outer')

        all_df_list.append(df_final)

    return all_df_list


def obs_data_loc():
    '''
    Outputs the sensor loction (x,y) on grid 
    '''
    sensor_pos = []
    channels = ['NO2', 'O3', 'PM10', 'PM25']
    for c in channels:
        with open("air_paris/new_observation_data/station_data_"+c+ ".pkl", "rb") as f:
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


class AirPollutionDataset(Dataset):
    def __init__(self, img_shape, mean_std_data=None, dict_min_max=None, sensor_pos=None, vt=True, df_list=[]):
        
        self.img_shape = img_shape
        self.sensor_pos = sensor_pos
        self.mean_std_data = mean_std_data
        self.dict_min_max = dict_min_max
        self.vt = vt
        self.df_list = df_list

    def __len__(self):
        return len(self.df_list[0])
    
    def normalize(self, img):
        C, H, W = img.shape
        for i in range(C):
            img[i] = 0.5*(img[i]-self.mean_std_data[i][0])/self.mean_std_data[i][1]

        return img
    
    def minmax(self, img):
        C, H, W = img.shape
        for i in range(C):
            img[i] = (img[i] - self.dict_min_max[i][0])/(self.dict_min_max[i][1] - self.dict_min_max[i][0])
        return img


    def voronoi_tess(self, mask, values_list):
        C, H, W = self.img_shape  #8518, 1, 75, 110
        grid_x, grid_y = np.mgrid[0:H, 0:W]

        new_img = np.zeros(self.img_shape)

        for channel in range(C):
            flatten_channel_idx, flatten_y_idx, flatten_x_idx = torch.nonzero(mask, as_tuple=True)
            target_idx = flatten_channel_idx == channel
            points = torch.column_stack((flatten_x_idx[target_idx], flatten_y_idx[target_idx]))

            new_img[channel] = griddata(points[:,[1,0]], values_list[channel], (grid_x, grid_y), method='nearest')

        return (torch.from_numpy(new_img)).to(torch.float32)
    
    def masking(self, sensor_obs_list, index):
        np.random.seed(index)
        C, H, W = self.img_shape  # channel, length, dim
        if self.sensor_pos:
                values_list = []
                mask = torch.zeros(self.img_shape)
                org_img = torch.zeros(self.img_shape)

                for i in range(C):

                    arr = self.sensor_pos[i]
                    keep_points = math.floor(arr.shape[0]*0.85)

                    arr = arr[sensor_obs_list[i] > 0.0]
                    values = sensor_obs_list[i][sensor_obs_list[i] > 0.0]
                    if self.mean_std_data:
                        values = 0.5*(values - self.mean_std_data[i][0])/self.mean_std_data[i][1]

                    for t, coord in enumerate(arr):
                        org_img[i,coord[1], coord[0]] = values[t]


                    must_keep_mask = ((arr[:, 0] < 20) | (arr[:, 0] > 80)) & (arr[:, 1] > 55)
                    must_keep = arr[must_keep_mask]
                    must_keep_values = values[must_keep_mask]
                    remaining = arr[~must_keep_mask]
                    remaining_values = values[~must_keep_mask]

                    num_to_sample = keep_points - len(must_keep)

                    if num_to_sample>len(remaining):
                        num_to_sample = len(remaining)

                    selected_additional = remaining[np.random.choice(len(remaining), size=num_to_sample, replace=False)]
                    selected_additional_values = remaining_values[np.random.choice(len(remaining), size=num_to_sample, replace=False)]

                    final_selected = np.concatenate([must_keep, selected_additional], axis=0)
                    final_selected_values = np.concatenate([must_keep_values, selected_additional_values], axis=0)

                    values_list.append(final_selected_values)

                    for x,y in final_selected:
                            mask[i,y,x] = 1
                
                return mask, values_list, org_img
        
        else:
            return None, None, None

    def __getitem__(self, index):
        sensor_obs_list = []
        for i in range(4):
            yo = self.df_list[i].loc[index][1:].to_numpy()
            yo = np.array([float(x) for x in yo])
            sensor_obs_list.append(yo)
        
        mask, values_list, org_img = self.masking(sensor_obs_list, index)

        if self.vt:
            vt = self.voronoi_tess(mask, values_list)
        else:
            vt = None

        if self.dict_min_max:
            vt = self.minmax(vt)

        return mask, vt, org_img


def load_data(img_shape, normalisation, df_obs_val_list=None):
    '''
    img_shape: (C,H,W) 
    normalisation: True if gaussian norm applied, False if min-max norm applied

    returns: Torch Dataset
    '''
    if df_obs_val_list is None:
        df_obs_val_list = create_df_list()
    sensor_pos = obs_data_loc()
    dict_minmax, dict_meanstd = data_stat()

    if normalisation:
        dataset_polair = AirPollutionDataset(img_shape=img_shape, mean_std_data=dict_meanstd, dict_min_max=None, sensor_pos=sensor_pos, vt=True, df_list=df_obs_val_list)
    else:
        dataset_polair = AirPollutionDataset(img_shape=img_shape, mean_std_data=None, dict_min_max=dict_minmax, sensor_pos=sensor_pos, vt=True, df_list=df_obs_val_list)

    return dataset_polair


