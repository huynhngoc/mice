import pandas as pd
import h5py
import numpy as np
from matplotlib import pyplot as plt
import os


base_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/'
dataset_filename = base_path + 'datasets/mice.h5'
model = 'b4_lr0005'

# merge interpret results
log_folders = [d for d in os.listdir(
    base_path + 'orion') if model in d and 'concat' not in d]


def merge_vargrad(fold):
    selected_folders = [d for d in log_folders if d.endswith(str(fold))]
    predicted = []
    tta = []
    for d in selected_folders:
        df = pd.read_csv(base_path + 'orion/' + d + '/mc_predicted.csv')
        predicted.append(df['predicted'])
        tta.append(df.values[:, 2:])

    imgs = []
    for d in selected_folders:
        with h5py.File(base_path + 'orion/' + d + '/test_vargrad.h5', 'r') as f:
            imgs.append(f['vargrad'][:])
    with h5py.File(base_path + 'orion/' + d + '/test_vargrad.h5', 'r') as f:
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


for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(0)):
    vmax = np.quantile(vargrad, 0.9999)
    vmin = np.quantile(vargrad, 0.)
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
    fig.savefig(base_path + '/interpretability/' +
                f'pid_{pid:02d}_slice_{sid:02d}.png')
    plt.close('all')


for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(1)):
    vmax = np.quantile(vargrad, 0.9999)
    vmin = np.quantile(vargrad, 0.)
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
    fig.savefig(base_path + '/interpretability/' +
                f'pid_{pid:02d}_slice_{sid:02d}.png')
    plt.close('all')


for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(2)):
    vmax = np.quantile(vargrad, 0.9999)
    vmin = np.quantile(vargrad, 0.)
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
    fig.savefig(base_path + '/interpretability/' +
                f'pid_{pid:02d}_slice_{sid:02d}.png')
    plt.close('all')


for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(3)):
    vmax = np.quantile(vargrad, 0.9999)
    vmin = np.quantile(vargrad, 0.)
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
    fig.savefig(base_path + '/interpretability/' +
                f'pid_{pid:02d}_slice_{sid:02d}.png')
    plt.close('all')


for pid, sid, y, predicted, tta, img,  vargrad in zip(*get_vargard(4)):
    vmax = np.quantile(vargrad, 0.9999)
    vmin = np.quantile(vargrad, 0.)
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
    fig.savefig(base_path + '/interpretability/' +
                f'pid_{pid:02d}_slice_{sid:02d}.png')
    plt.close('all')
