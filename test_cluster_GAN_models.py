import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import *
from GAN_model import Generator, TrainSet
from torch.utils.data import DataLoader


def generate_data(context_a_data, context_b_data, gen_a, gen_b):
    context_a_is_weekend = True if context_a in [2, 5] else False
    context_b_is_weekend = True if context_b in [2, 5] else False

    data_a = []
    data_b = []
    if not context_a_is_weekend and not context_b_is_weekend:  # weekday to weekday
        temp_a = min([len(context_a_data[i]) for i in range(5)])
        temp_b = min([len(context_b_data[i]) for i in range(5)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(5):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    elif not context_a_is_weekend and context_b_is_weekend:  # weekday to weekend
        temp_a = min([len(context_a_data[i]) for i in range(2)])  # 周一周二 -> 周六周日
        temp_b = min([len(context_b_data[i]) for i in range(2)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(2):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    elif context_a_is_weekend and not context_b_is_weekend:  # weekend to weekday
        temp_a = min([len(context_a_data[i]) for i in range(5)])
        temp_b = min([len(context_b_data[i]) for i in range(5)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(5):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    else:  # weekend to weekend
        temp_a = min([len(context_a_data[i]) for i in range(2)])
        temp_b = min([len(context_b_data[i]) for i in range(2)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(2):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    test_dataset = TrainSet(data_a, data_b)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # start testing
    fake = []
    real = []
    original = []
    for x_a, x_b in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())
        real.append(x_b.detach().numpy())
        original.append(x_a.detach().numpy())

    # data processing
    fake_data = []
    real_data = []
    original_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])
            real_data.append(real[i][j])
            original_data.append(original[i][j])

    final_fake_data = []
    final_real_data = []
    final_original_data = []
    for i in range(len(real_data)):
        for j in range(real_data[i].shape[0]):
            final_real_data.append(real_data[i][j])
            final_fake_data.append(fake_data[i][j])
            final_original_data.append(original_data[i][j])

    return final_fake_data, final_real_data, final_original_data


def UNIT_Test_cl():
    # load data and set data loader
    with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(test_building_name, test_context_a, test_context_b), 'rb') as r:
        save_dict = pickle.load(r)

    context_a_data = save_dict.get('context_a_data')
    context_b_data = save_dict.get('context_b_data')
    load_max = save_dict.get('load_max')
    load_min = save_dict.get('load_min')

    # load cluster models
    with open('./models/{}_ashrae_{}_to_{}_cluster_models_dict.pkl'.format(test_model_name, test_context_a, test_context_b), 'rb') as r:
        models_dict = pickle.load(r)

    for cluster_id in range(n_clusters):
        this_gen_a = models_dict.get(cluster_id)[0]
        this_gen_b = models_dict.get(cluster_id)[1]

        if this_gen_a is None or this_gen_b is None:
            print('cluster {}, invalid model'.format(cluster_id))
            continue

        fake_data, real_data, original_data = generate_data(context_a_data=context_a_data,
                                                            context_b_data=context_b_data,
                                                            gen_a=this_gen_a,
                                                            gen_b=this_gen_b)

        # denormalize
        final_fake_data = [fake_data[i] * (load_max - load_min) + load_min for i in range(len(fake_data))]
        final_real_data = [real_data[i] * (load_max - load_min) + load_min for i in range(len(real_data))]
        final_original_data = [original_data[i] * (load_max - load_min) + load_min for i in range(len(original_data))]

        # error
        mae_list = [abs(final_real_data[i] - final_fake_data[i]) for i in range(len(final_real_data))]
        mae = sum(mae_list) / len(mae_list)
        print('cluster {}, mae = {}'.format(cluster_id, mae))

        # draw graphs
        fig = plt.figure(figsize=(10, 6))
        fig.add_subplot(111)
        plt.plot(range(len(final_real_data)), final_real_data, label='real_data', color='blue')
        plt.plot(range(len(final_fake_data)), final_fake_data, label='generated_data', color='orange')
        plt.plot(range(len(final_original_data)), final_original_data, label='original_data', color='green')
        plt.title('MAE = {}'.format(mae), loc='right')
        plt.title('{}_coolingLoad'.format(test_building_name))
        plt.grid()
        plt.legend(loc=1, fontsize=15)
        plt.show()


if __name__ == '__main__':
    UNIT_Test_cl()

