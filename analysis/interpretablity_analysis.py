import pandas as pd
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os


base_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/'
dataset_filename = base_path + 'datasets/mice_2025.h5'
interpretability_path = base_path + 'interpretability_results/'
model = 'b3_lr0001'


def merge_vargrad(fold, suffix='_02'):
    selected_folders = [d for d in os.listdir(interpretability_path) if d.endswith(str(fold))]
    predicted = []
    imgs = []
    tta = []
    for d in selected_folders:
        with h5py.File(interpretability_path + d + f'/test_vargrad{suffix}.h5', 'r') as f:
            imgs.append(f['vargrad'][:])
        df = pd.read_csv(interpretability_path + d + f'/tta_predicted{suffix}.csv')
        predicted.append(df['predicted'])
        tta.append(df.values[:, 2:])
    with h5py.File(interpretability_path + d + f'/test_vargrad{suffix}.h5', 'r') as f:
        pids = f['patient_idx'][:]

    imgs = np.mean(imgs, axis=0)

    return pids, np.mean(predicted, axis=0), np.concatenate(tta, axis=-1), imgs


def get_vargard(fold):
    with h5py.File(dataset_filename, 'r') as f:
        imgs = f[f'fold_{fold}']['x'][..., 0]
        sids = f[f'fold_{fold}']['slice_idx'][:]
        y = f[f'fold_{fold}']['y'][:]

    pids, predicted, tta, vargrads = merge_vargrad(fold)
    return pids, sids, y, predicted, tta, imgs, vargrads

suffix = '_02'

for fold in range(1, 5):
    for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(fold)):
        vmax = np.quantile(vargrad, 0.9999)
        vmin = np.quantile(vargrad, 0.0001)
        thres = np.quantile(vargrad, 0.85)
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(img, 'gray')
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(vargrad, 'hot', vmin=vmin, vmax=vmax)

        explain_map = vargrad.copy()
        explain_map[explain_map < thres] = np.nan
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(img, 'gray')
        plt.imshow(explain_map, 'hot', alpha=0.5, vmin=vmin, vmax=vmax)

        # plt.subplot(1,4,4)
        # plt.violinplot(tta)

        plt.suptitle(
            f'PID: {pid}, slice {sid}, class {y}, predicted {predicted:.3f}')
        plt.subplots_adjust(wspace=0.01, left=0.001, right=0.9999)
        # plt.show()
        fig.savefig(base_path + f'interpretability_vis/interpretability{suffix}/' +
                    f'pid_{pid:02d}_slice_{sid:02d}.png')
        plt.close('all')



suffix = '_05'

for fold in range(5):
    for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(fold)):
        vmax = np.quantile(vargrad, 0.9999)
        vmin = np.quantile(vargrad, 0.0001)
        thres = np.quantile(vargrad, 0.85)
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(img, 'gray')
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(vargrad, 'hot', vmin=vmin, vmax=vmax)

        explain_map = vargrad.copy()
        explain_map[explain_map < thres] = np.nan
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(img, 'gray')
        plt.imshow(explain_map, 'hot', alpha=0.5, vmin=vmin, vmax=vmax)

        # plt.subplot(1,4,4)
        # plt.violinplot(tta)

        plt.suptitle(
            f'PID: {pid}, slice {sid}, class {y}, predicted {predicted:.3f}')
        plt.subplots_adjust(wspace=0.01, left=0.001, right=0.9999)
        # plt.show()
        fig.savefig(base_path + f'interpretability_vis/interpretability{suffix}/' +
                    f'pid_{pid:02d}_slice_{sid:02d}.png')
        plt.close('all')
