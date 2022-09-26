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


# note: 这里移除了原先的“chillerName”，因为dataloder做出来的数据是所有chiller整合起来的数据
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


if __name__ == '__main__':
    # 1. load csv data
    with open('./tmp_pkl_data/{}_data_dict.pkl'.format(train_model_name), 'rb') as r:
        save_dict = pickle.load(r)

    data = save_dict.get('df')

    # 1.2 create category column
    data = build_categories_columns(df=data)
    print('Done 1. load csv data')

    # 2. build data into tasks
    """
    * task_dict and metadata_task_list length are same, of course.

    1. task_dict：
    {
    metadata: df_small,  # e.g., (1, 1, 'autumn', 'midnight', 2017): df_small
    ...
    }: 

    2. metadata_task_list:
    [
    (1, 1, 'autumn', 'evening', 2017), 
    (1, 1, 'autumn', 'midnight', 2017), 
    (1, 1, 'spring', 'evening', 2017)
    ]

    """
    task_dict = {}
    metadata_task_list = []
    for metadata, df_small in data.groupby(categories_columns_name_list):
        task_dict.update({metadata: df_small})
        metadata_task_list.append(metadata)
    print('Done 2. build data into %s tasks' % metadata_task_list.__len__())

    # 3. build matrix
    d_mate_matrix = build_matrix_by_metadata(metadata_task_list=metadata_task_list)
    print('Done 3. build matrix')

    # 4. meta-clustering modeling
    """
    metadata_clustering_dict = {
    0: [3, 5, 43],
    1: [...]
    }
    """
    metadata_clustering_dict = {}
    # X = np.array([[1, 1], [2, 1], [1, 0],
    #               [4, 7], [3, 5], [3, 6]])
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels='discretize',
                                    random_state=0,
                                    affinity='precomputed').fit(d_mate_matrix)
    labels = clustering.labels_
    for c in range(n_clusters):
        metadata_clustering_dict.update({c: np.where(labels == c)[0]})

    # 5. train model for all clustering (attention: not for metadata, it's cluster)
    # note: context switching parameters
    metadata_clustering_models_dict = {}
    for cluster_id, metadata_id_list in metadata_clustering_dict.items():
        print('Train model for cluster {}'.format(cluster_id))
        df_this_cluster_list = []
        for metadata_id in metadata_id_list:
            this_metadata = metadata_task_list[metadata_id]
            df_this_cluster_list.append(task_dict[this_metadata])
        df_this_cluster = pd.concat(df_this_cluster_list, axis=0).reset_index(drop=True, inplace=False)

        cl_a, cl_b, _, _ = create_a_and_b(df=df_this_cluster, context_a=context_a, context_b=context_b)
        if cl_a.shape[0] == 0 or cl_b.shape[0] == 0:
            gen_a, gen_b = None, None
        else:
            gen_a, gen_b = UNIT_Train_cl(data_a=cl_a, data_b=cl_b)
        metadata_clustering_models_dict.update({cluster_id: (gen_a, gen_b)})
        print('model saved')

    with open('models/{}_ashrae_{}_to_{}_cluster_models_dict.pkl'.format(train_model_name, context_a, context_b), 'wb') as w:
        pickle.dump(metadata_clustering_models_dict, w)










    # 5. inference and evaluation
    # total_error = 0
    # sample_num = 0
    # check_clustering_index = 0
    # cluster = metadata_clustering_dict[check_clustering_index]
    # print('this cluster: {}'.format(cluster))
    #
    # for choose_i in range(cluster.shape[0]):
    #     df_to_infer = None
    #     df_train_from_this_cluster = []
    #
    #     for i, metadata_index in enumerate(cluster):
    #         the_metadata_tuple = metadata_task_list[metadata_index]
    #         df_by_metadata = task_dict[the_metadata_tuple]
    #
    #         if i == choose_i:
    #             df_to_infer = df_by_metadata.reset_index(drop=True)
    #         else:
    #             df_train_from_this_cluster.append(df_by_metadata)
    #
    #     df_train = pd.concat(df_train_from_this_cluster, axis=0).reset_index(drop=True)
    #
    #     # 5.2 train and evaluate
    #     train_save_dict = read_data(data=df_train, cluster=check_clustering_index, context_a=context_a, context_b=context_b)
    #     test_save_dict = read_data(data=df_to_infer, cluster=check_clustering_index, context_a=context_a, context_b=context_b)
    #
    #     gen_a, gen_b = UNIT_Train_cl(cluster=check_clustering_index, context_a=context_a, context_b=context_b, save_dict=train_save_dict)
    #     error = UNIT_Test_cl(cluster=check_clustering_index, context_a=context_a, context_b=context_b,
    #                          save_dict=test_save_dict, gen_a=gen_a, gen_b=gen_b)
    #
    #     total_error += error
    #     sample_num += df_to_infer.shape[0]
    #
    # mean_error_this_cluster = total_error / sample_num
    # print('error for cluster %s:' % check_clustering_index, mean_error_this_cluster)
    # print('sample num:', sample_num)

    # 6.
    # 6.1 get all task/sample in this cluster
    # df_all_this_cluster_list = []
    # for i, metadata_index in enumerate(cluster):
    #     the_metadata_tuple = metadata_task_list[metadata_index]
    #     df_by_metadata = task_dict[the_metadata_tuple]
    #
    #     df_all_this_cluster_list.append(df_by_metadata)
    # df_all_this_cluster = pd.concat(df_all_this_cluster_list).reset_index(drop=True)
    #
    # # 6.2 用其余所有的数据作为tran data
    # print('shape:', df_all_this_cluster.shape, data.shape)
    # data = data.append(df_all_this_cluster).drop_duplicates(keep=False)
    # print('shape:', df_all_this_cluster.shape, data.shape)
    #
    # # 6.3 train and evaluate
    # train_save_dict = read_data(data=data, cluster=check_clustering_index, context_a=context_a, context_b=context_b)
    # test_save_dict = read_data(data=df_all_this_cluster, cluster=check_clustering_index, context_a=context_a,
    #                            context_b=context_b)
    #
    # gen_a, gen_b = UNIT_Train_cl(cluster=check_clustering_index, context_a=context_a, context_b=context_b,
    #                              save_dict=train_save_dict)
    # error = UNIT_Test_cl(cluster=check_clustering_index, context_a=context_a, context_b=context_b,
    #                      save_dict=test_save_dict, gen_a=gen_a, gen_b=gen_b)
    # print('error for cluster %s:' % check_clustering_index, error.mean())

