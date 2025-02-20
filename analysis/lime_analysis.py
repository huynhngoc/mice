import os
import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


base_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/'
# dataset_filename = base_path + 'datasets/mice_2025.h5'
dataset_filename = '../mice_2025.h5'
interpretability_path = base_path + 'interpretability_results/'
model = 'b3_lr0001'


def merge_lime(fold):
    selected_folders = [d for d in os.listdir(interpretability_path) if d.endswith(str(fold))]
    predicted = []
    imgs = []
    tta = []
    for d in selected_folders:
        with h5py.File(interpretability_path + d + f'/test_lime.h5', 'r') as f:
            imgs.append(f['lime'][:])
        df = pd.read_csv(interpretability_path + d + f'/tta_predicted_05.csv')
        predicted.append(df['predicted'])
        tta.append(df.values[:, 3:])
    with h5py.File(interpretability_path + d + f'/test_lime.h5', 'r') as f:
        pids = f['patient_idx'][:]

    # imgs = np.mean(imgs, axis=0)

    return pids, np.stack(predicted, axis=-1), np.concatenate(tta, axis=-1), np.stack(imgs, axis=-1)


def get_lime(fold):
    with h5py.File(dataset_filename, 'r') as f:
        imgs = f[f'fold_{fold}']['x'][..., 0]
        sids = f[f'fold_{fold}']['slice_idx'][:]
        y = f[f'fold_{fold}']['y'][:]

    pids, predicted, tta, limes = merge_lime(fold)
    return pids, sids, y, predicted, tta, imgs, limes



for fold in range(5):
    for pid, sid, y, predicted, tta, img,  lime in zip(*get_lime(fold)):
        # vmax = np.quantile(vargrad, 0.9999)
        # vmin = np.quantile(vargrad, 0.0001)
        # thres = np.quantile(vargrad, 0.85)
        fig = plt.figure(figsize=(13, 8))
        plot_idx = 1
        vmin, vmax = -abs(lime).max(), abs(lime).max()
        thres = 0.1
        for lime_img in np.transpose(lime, (2, 0, 1)):
            # vmin, vmax = -abs(lime_img).max(), abs(lime_img).max()
            # thres_pos = np.quantile(lime_img[lime_img > 0], 0.85)
            # thres_neg = np.quantile(lime_img[lime_img < 0], 0.85)
            # thres = max(thres_pos, -thres_neg)
            plt.subplot(3, 5, plot_idx)
            plt.axis('off')
            plt.imshow(img, 'gray')
            plt.title(f'P {plot_idx}: {predicted[plot_idx - 1]:.3f}')
            plt.subplot(3, 5, plot_idx + 5)
            plt.axis('off')
            plt.imshow(lime_img, 'RdBu_r', vmin=vmin, vmax=vmax)

            explain_map = lime_img.copy()
            explain_map[(-thres < lime_img) & (lime_img < thres)] = np.nan
            plt.subplot(3, 5, plot_idx + 10)
            plt.axis('off')
            plt.imshow(img, 'gray')
            plt.imshow(explain_map, 'RdBu_r', alpha=0.5, vmin=vmin, vmax=vmax)
            plot_idx += 1

        lime_avg = lime.mean(axis=-1)
        # vmin, vmax = -abs(lime_avg).max(), abs(lime_avg).max()
        # thres_pos = np.quantile(lime_avg[lime_avg > 0], 0.85)
        # thres_neg = np.quantile(lime_avg[lime_avg < 0], 0.85)
        # thres = max(thres_pos, -thres_neg)
        plt.subplot(3, 5, plot_idx)
        plt.axis('off')
        plt.imshow(img, 'gray')
        plt.title(f'Avg: {predicted.mean(axis=-1):.3f}')
        plt.subplot(3, 5, plot_idx + 5)
        plt.axis('off')
        plt.imshow(lime_avg, 'RdBu_r', vmin=vmin, vmax=vmax)

        explain_map = lime_avg.copy()
        explain_map[(-thres < lime_avg) & (lime_avg < thres)] = np.nan
        plt.subplot(3, 5, plot_idx + 10)
        plt.axis('off')
        plt.imshow(img, 'gray')
        im = plt.imshow(explain_map, 'RdBu_r', alpha=0.5, vmin=vmin, vmax=vmax)
        im.set_clim(vmin, vmax)
        cb = fig.colorbar(im, ax=fig.get_axes(), orientation='vertical', fraction=0.046, pad=0.01)
        cb.set_label('LIME')

        plt.suptitle(
            f'PID: {pid}, slice {sid}, class {y}')
        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.001, right=0.85)
        # plt.show()
        fig.savefig(base_path + f'interpretability_vis/lime/' +
                    f'pid_{pid:02d}_slice_{sid:02d}.png')
        plt.close('all')
