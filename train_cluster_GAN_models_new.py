# -*- coding: utf-8 -*-

"""
1. 选定一个target building和一个context转换的情景（context 1 -> 4）
2. 手上有的数据就是target building的context 1 以及其他楼的context 1 和 4 的数据
3. 要做的就是从其他楼的context 1 和 4 中筛选出能train出对于target building效果最好的GAN的数据
4. 目前能做的就是用target building的context 1 对其他楼的context 1 数据进行筛选（聚类 + 相关性）
"""
import pandas as pd
import numpy as np
import pickle

from config import *
from sklearn.cluster import SpectralClustering
from UNIT_HKisland_with_cluster.dataloader_for_multiple_buildings import create_a_and_b
from UNIT_HKisland_with_cluster.GAN_train_for_cluster import UNIT_Train_cl


categories_columns_name_list = [
    'outdoor_temp_level',
    'season_level',
    'hour_level',
    'year'
]


def outdoor_temp_map(x):
    if x < 10:
        outdoor_temp_index = 0
    elif x < 20:
        outdoor_temp_index = 1
    elif x < 30:
        outdoor_temp_index = 2
    else:
        outdoor_temp_index = 3
    return outdoor_temp_index


def time_hour_map(x):
    if 0 <= x <= 6:
        hour_index = 'midnight'
    elif 7 <= x <= 11:
        hour_index = 'morning'
    elif 12 <= x <= 18:
        hour_index = 'afternoon'
    else:
        hour_index = 'evening'
    return hour_index


def time_year_map(x):
    pass


def time_month_map(x):
    if 3 <= x <= 5:
        month_index = 'spring'
    elif 6 <= x <= 9:
        month_index = 'summer'
    elif 10 <= x <= 11:
        month_index = 'autumn'
    else:
        month_index = 'winter'
    return month_index


def build_categories_columns(df):
    # 1. add time related columns
    df['time'] = df['time'].astype("datetime64")
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour

    # map to metadata category level
    df["outdoor_temp_level"] = df["temperature"].map(outdoor_temp_map)
    df["season_level"] = df["month"].map(time_month_map)
    df["hour_level"] = df["hour"].map(time_hour_map)
    return df


def build_matrix_by_metadata(metadata_task_list):
    tasks_num = metadata_task_list.__len__()
    d_meta = np.zeros((tasks_num, tasks_num))

    # stage II: Jaccard Distance
    for i, metadata_1 in enumerate(metadata_task_list):
        for j, metadata_2 in enumerate(metadata_task_list):
            diff_count = 0  # metadata 不一样的数量
            for ii in range(len(categories_columns_name_list)):
                diff_count += (metadata_1[ii] != metadata_2[ii])
            j_d = 1 - (len(categories_columns_name_list) - diff_count) / (
                        len(categories_columns_name_list) + diff_count)
            d_meta[i, j] = j_d

    return d_meta


def kendall_correlation_coefficient(arr_1, arr_2):
    series_1 = pd.Series(arr_1)
    series_2 = pd.Series(arr_2)
    return series_1.corr(series_2, method='kendall')


if __name__ == '__main__':
    # 1. load data
    # 1.1 get context a and context b data of each training building
    each_train_building_context_a_dict = {}
    each_train_building_context_b_dict = {}
    data_length_list = []
    for building_name in multiple_building_names:
        with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(building_name, context_a, context_b), 'rb') as r:
            this_save_dict = pickle.load(r)
        this_df = this_save_dict.get('df')
        this_context_a_data = this_df[this_df['8_class'] == context_a].reset_index(drop=True, inplace=True)
        this_context_b_data = this_df[this_df['8_class'] == context_b].reset_index(drop=True, inplace=True)
        each_train_building_context_a_dict.update({building_name: this_context_a_data})
        each_train_building_context_b_dict.update({building_name: this_context_b_data})
        data_length_list.append(this_context_a_data.shape[0])

    # 1.2 get context a data of target building
    with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(target_building_name, context_a, context_b), 'rb') as r:
        target_save_dict = pickle.load(r)
    target_building_df = target_save_dict.get('df')
    target_building_context_a_data = target_building_df[target_building_df['8_class'] == context_a].reset_index(drop=True, inplace=True)
    data_length_list.append(target_building_context_a_data.shape[0])

    # 2. use kendall correlation coefficient to find useful training buildings
    # 2.1 get min length among all context a data
    min_length = min(data_length_list)

    # 2.2 calculate coefficient between each training building and target building
    useful_context_a_train_building = []
    target_context_a_cl = target_building_context_a_data.loc[:min_length, 'nor_cl']
    for train_building in each_train_building_context_a_dict.keys():
        train_context_a_cl = each_train_building_context_a_dict.get(train_building).loc[:min_length, 'nor_cl']
        this_kendall_coefficient = kendall_correlation_coefficient(target_context_a_cl, train_context_a_cl)

        # set a threshold value
        # if kendall coefficient is larger than the value, we consider this building useful for the target building
        threshold_val = 0.5
        if this_kendall_coefficient >= 0.5:
            useful_context_a_train_building.append(train_building)

    # 3. use clustering within each useful train building to further select valuable data
    aaa = 1










