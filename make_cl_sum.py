# -*- coding: utf-8 -*-

"""

@file: make_cl_sum.py
@time: 2021/8/20 20:59
@desc:

chiller 的总load求和
"""

import pandas as pd
pd.options.display.max_columns = None


def make_big_df(data):
    # 1. load csv data
    # data = pd.read_csv('data_27012021/spot_objs20210127.csv', header=None, index_col=0)
    print(data.shape)
    print('chillerName:', data['chillerName'].unique())
    data = data.loc[data['cop'] > 0, :].reset_index(drop=True)
    print('过滤cop=0的data, shape:', data.shape)

    # data = data.loc[(data['cop'] >= 1) & (data['cop'] <= 10)].reset_index(drop=True)
    # print('过滤cop太小的data, shape:', data.shape)

    # plt.hist(data.cop, 50)
    # plt.show()
    # plt.hist(data.supplyTemp, 50)
    # plt.show()
    # plt.hist(data.flowRate, 50)
    # plt.show()

    data.reset_index(drop=True, inplace=True)
    print(data.shape)
    count = data['time'].value_counts()
    print('max run simultaneously:', count[0])

    # data['time_stamp'] = pd.to_datetime(data['time'], unit='s')

    # 2. get each chiller data, and adjust columns naming
    chillers_list = []
    for i in data['chillerName'].unique():
        data_i = data[data['chillerName'] == i]
        if data_i.shape[0] <= 0:
            break
        print('i=', i, 'data_i.shape = ', data_i.shape)
        columns = data_i.columns.to_list()
        for j in range(2, len(columns)):
            columns[j] = columns[j] + '_' + str(i)
        data_i.columns = columns
        chillers_list.append(data_i)

    # 3. 把上面的的子表全部拼起来
    df_big = chillers_list[0]
    for i in range(len(chillers_list) - 1):
        df_big = pd.merge(df_big, chillers_list[i + 1], on=['time', 'building'], how='outer')

    df_big.sort_values('time', inplace=True)
    df_big.reset_index(drop=True, inplace=True)
    print('after merge by time:', df_big.shape)


    # 对每行求load总和
    df_big['coolingLoad'] = 0
    load_columns_name_list = [item for item in df_big.columns.to_list() if 'coolingLoad' in item]
    df_loads = df_big[load_columns_name_list]
    df_loads = df_loads.fillna(0)
    df_big['coolingLoad'] = df_loads.sum(axis=1)

    df_big['temperature'] = 0
    temp_columns_name_list = [item for item in df_big.columns.to_list() if 'temperature' in item]
    df_temp = df_big[temp_columns_name_list]
    df_temp = df_temp.fillna(0)
    df_big['temperature'] = df_temp.max(axis=1)

    return df_big[['time', 'coolingLoad', 'temperature']]

# def make_big_df_with_all_features(path)

if __name__ == '__main__':
    # find_chiller_cl_sum()

    path = 'raw_data/CP4.csv'
    make_big_df(path)
