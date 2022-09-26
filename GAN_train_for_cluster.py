from GAN_model import Generator, Discriminator, TrainSet
from torch.utils.data import DataLoader
from config import *

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


def check_training_result(gen_a, gen_b, cluster, context_a, context_b, save_dict):
    context_a_is_weekend = True if context_a in [2, 5] else False
    context_b_is_weekend = True if context_b in [2, 5] else False

    # get domain a and domain b data
    context_a_data = save_dict.get('context_a_data')
    context_b_data = save_dict.get('context_b_data')
    load_max = save_dict.get('load_max')
    load_min = save_dict.get('load_min')

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
    print('test on: cluster_{}_ashrae_{}_to_{}_cl.pkl'.format(cluster, context_a, context_b))
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

    # denormalize
    denorm_fake_data = [final_fake_data[i] * (load_max - load_min) + load_min for i in range(len(final_fake_data))]
    denorm_real_data = [final_real_data[i] * (load_max - load_min) + load_min for i in range(len(final_real_data))]
    denorm_original_data = [final_original_data[i] * (load_max - load_min) + load_min for i in range(len(final_original_data))]

    # error
    mae_list = [abs(denorm_real_data[i] - denorm_fake_data[i]) for i in range(len(denorm_real_data))]
    mae = sum(mae_list) / len(mae_list)

    # draw graphs
    # fig = plt.figure(figsize=(10, 6))
    # fig.add_subplot(111)
    # plt.plot(range(len(denorm_real_data)), denorm_real_data, label='real_data', color='blue')
    # plt.plot(range(len(denorm_fake_data)), denorm_fake_data, label='fake_data', color='orange')
    # plt.plot(range(len(denorm_original_data)), denorm_original_data, label='original_data', color='green')
    # plt.title('MAE = {}'.format(mae), loc='right')
    # plt.title('{}_coolingLoad'.format(train_building_name))
    # plt.grid()
    # plt.legend()
    # plt.savefig('./train_result/{}_epoch_{}.png'.format(train_building_name, epoch))

    return mae


def UNIT_Train_cl(data_a, data_b):
    # initialize trained_models
    gen_a = Generator()
    gen_b = Generator()
    dis_a = Discriminator()
    dis_b = Discriminator()

    # set up data loader
    train_dataset = TrainSet(data_a, data_b)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # set up parameters
    gen_params = list(gen_a.parameters()) + list(gen_b.parameters())
    dis_params = list(dis_a.parameters()) + list(dis_b.parameters())
    gen_opt = torch.optim.Adam(gen_params, lr=gen_learning_rate)
    dis_opt = torch.optim.Adam(dis_params, lr=dis_learning_rate)

    # start training
    for epoch in range(num_epochs):
        gen_loss_sum = 0
        dis_loss_sum = 0
        for i, (x_a, x_b) in enumerate(train_loader):
            x_a = x_a.to(torch.float32)
            x_b = x_b.to(torch.float32)

            # train discriminator
            dis_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # discriminator loss
            loss_dis_a = dis_a.cal_dis_loss(x_ba, x_a)
            loss_dis_b = dis_b.cal_dis_loss(x_ab, x_b)
            loss_dis_all = loss_dis_a + loss_dis_b
            dis_loss_sum += loss_dis_all
            loss_dis_all.backward()
            dis_opt.step()

            # train generator
            gen_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = gen_a.decode(h_a + n_a)
            x_b_recon = gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # encode again
            h_b_recon, n_b_recon = gen_a.encode(x_ba)
            h_a_recon, n_a_recon = gen_b.encode(x_ab)
            # decode again
            x_bab = gen_b.decode(h_b_recon + n_b_recon)
            x_aba = gen_a.decode(h_a_recon + n_a_recon)
            # generator loss
            # reconstruction loss
            loss_gen_recon_x_a = torch.mean(torch.abs(x_a_recon - x_a))
            loss_gen_recon_x_b = torch.mean(torch.abs(x_b_recon - x_b))
            # GAN loss
            loss_gen_adv_a = dis_a.cal_gen_loss(x_ba)
            loss_gen_adv_b = dis_b.cal_gen_loss(x_ab)
            # cycle-consistency loss
            loss_gen_cycle_cons_a = torch.mean(torch.abs(x_aba - x_a))
            loss_gen_cycle_cons_b = torch.mean(torch.abs(x_bab - x_b))
            # total loss
            loss_gen_all = loss_gen_recon_x_a + loss_gen_recon_x_b + \
                           loss_gen_adv_a + loss_gen_adv_b + \
                           loss_gen_cycle_cons_a + loss_gen_cycle_cons_b
            gen_loss_sum += loss_gen_all
            loss_gen_all.backward()
            gen_opt.step()

        print('Epoch {}: Generator loss: {} \t Discriminator loss: {}'.format(epoch + 1, gen_loss_sum, dis_loss_sum))

    return gen_a, gen_b

