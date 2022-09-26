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
from dataloader_for_multiple_buildings import create_a_and_b
from GAN_train_for_cluster import UNIT_Train_cl

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


def DTW_distance(arr_1, arr_2, max_warping_window=10000):
    """
    source: https://www.cnblogs.com/ningjing213/p/10502519.html
    :param arr_1
    :param arr_2
    :param max_warping_window
    :return:
    """
    ts_a = np.array(arr_1)
    ts_b = np.array(arr_2)
    M = ts_a.shape[0]
    N = ts_b.shape[0]
    cost = np.ones((M, N))
    d = lambda x, y: ((x - y) ** 2)

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


def get_fixed_length_data(data, num_unit, context_is_weekend):
    output = np.array([])

    if context_is_weekend:
        for i in range(num_unit):
            for j in range(2):
                output = np.append(output, data[j][i])
    else:
        for i in range(num_unit):
            for j in range(5):
                output = np.append(output, data[j][i])

    return output


if __name__ == '__main__':
    # 1. load data
    # 1.1 get context a and context b data of each training building
    each_train_building_context_a_dict = {}
    each_train_building_context_b_dict = {}
    unit_count_list = []
    for building_name in multiple_building_names:
        with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(building_name, context_a, context_b),
                  'rb') as r:
            this_save_dict = pickle.load(r)
        this_context_a_data = this_save_dict.get('context_a_data')
        this_context_b_data = this_save_dict.get('context_b_data')
        each_train_building_context_a_dict.update({building_name: this_context_a_data})
        each_train_building_context_b_dict.update({building_name: this_context_b_data})

        unit_count_list.append(min([len(this_context_a_data[i]) for i in range(len(this_context_a_data))]))

    # 1.2 get context a data of target building
    with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(target_building_name, context_a, context_b),
              'rb') as r:
        target_save_dict = pickle.load(r)
    target_building_context_a_data = target_save_dict.get('context_a_data')

    unit_count_list.append(min([len(target_building_context_a_data[i])
                                for i in range(len(target_building_context_a_data))]))

    # 2. use kendall correlation coefficient to find useful training buildings
    # 2.1 get min number of unit
    num_of_units = min(unit_count_list)

    # 2.2 get fixed length data
    fixed_length_data_for_train_building_dict = {}
    for building_name in each_train_building_context_a_dict.keys():
        this_fixed_data = get_fixed_length_data(data=each_train_building_context_a_dict.get(building_name),
                                                num_unit=num_of_units,
                                                context_is_weekend=context_a_is_weekend)
        fixed_length_data_for_train_building_dict.update({building_name: this_fixed_data})

    fixed_length_data_for_target_building = get_fixed_length_data(data=target_building_context_a_data,
                                                                  num_unit=num_of_units,
                                                                  context_is_weekend=context_a_is_weekend)

    # 2.3 calculate coefficient between each training building and target building
    useful_context_a_train_building = []
    for building_name in fixed_length_data_for_train_building_dict.keys():
        evaluation_metrics = DTW_distance(arr_1=fixed_length_data_for_target_building,
                                          arr_2=fixed_length_data_for_train_building_dict.get(building_name))

        print('{}, DTW_Distance: {}'.format(building_name, evaluation_metrics))

        # set a threshold value
        # if kendall coefficient is larger than the value, we consider this building useful for the target building
        # threshold_val = 0.6
        # if this_kendall_coefficient >= threshold_val:
        #     useful_context_a_train_building.append(building_name)

    # 3. use clustering within each useful train building to further select valuable data
