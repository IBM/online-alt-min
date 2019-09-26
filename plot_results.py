import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_stat_data_files(dirname, EXPS, field, lab=''):
    all_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    files = [f for f in all_files if lab in f]

    if type(field) is not list: field = [field]

    field_m, field_sd = [], []
    for exp in EXPS:
        exp_data = []
        exp_files = [f for f in files if exp in f]
        for f in exp_files:
            recursive_field = torch.load(os.path.join(dirname, f))
            for l in field:
                recursive_field = recursive_field[l]
            exp_data.append(recursive_field)
        exp_data = np.r_[exp_data]
        field_m.append(exp_data.mean(0))
        field_sd.append(exp_data.std(0))

    field_m = np.r_[field_m].T
    field_sd = np.r_[field_sd].T

    return field_m, field_sd



DIR = './output'
EXPS = ['adam_baseline', 'altmin_sgd']

# Plot 1 to 4
for TASK in ['mnist', 'cifar10']:
    for TITLE in ['feedforward_2x100_' + TASK, 'feedforward_2x500_' + TASK]:
        perf_te, _ = get_stat_data_files(DIR, EXPS, ['perf', 'te'], TITLE)
        perf_te_first, _ = get_stat_data_files(DIR, EXPS, ['perf', 'first_epoch'], TITLE)

        plt.figure(figsize=(12, 4))

        ax1 = plt.subplot(1, 2, 1)
        plt.plot(perf_te_first[:50], linewidth=2, marker='.')
        plt.legend(EXPS)
        plt.ylabel('test accuracy')
        plt.xlabel('mini-batch')
        plt.title(TITLE)

        ax2 = plt.subplot(1, 2, 2)
        plt.plot(perf_te, linewidth=2, marker='.')
        plt.legend(EXPS)
        plt.ylabel('test accuracy')
        plt.xlabel('epoch')


# Plot lenet on mnist
TITLE = 'lenet_mnist'
perf_te, _ = get_stat_data_files(DIR, EXPS, ['perf', 'te'], TITLE)

plt.figure(figsize=(6, 4))

plt.plot(perf_te, linewidth=2, marker='.')
plt.legend(EXPS)
plt.ylabel('test accuracy')
plt.xlabel('mini-batch')
plt.title(TITLE)


# higgs dataset
TITLE = 'feedforward_2x300_higgs'
perf_te, _ = get_stat_data_files(DIR, EXPS, ['perf', 'te_vs_iterations'], TITLE)

plt.figure(figsize=(6, 4))

plt.plot(np.arange(200, 200*(len(perf_te)+1), 200), perf_te, linewidth=2, marker='.')
plt.legend(EXPS)
plt.ylabel('test accuracy')
plt.xlabel('mini-batch')
plt.title(TITLE)


# Plot BinaryNet
TITLE = 'binary_2x500'
perf_te, _ = get_stat_data_files(DIR, ['altmin_sgd'], ['perf', 'te'], TITLE)
perf_tr, _ = get_stat_data_files(DIR, ['altmin_sgd'], ['perf', 'tr'], TITLE)

plt.figure(figsize=(6, 4))

plt.plot(perf_tr, linewidth=2, marker='.', label='training accuracy')
plt.plot(perf_te, linewidth=2, marker='.', label='test accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title(TITLE)
