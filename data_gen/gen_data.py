import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from deoxys_image import normalize
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold
import h5py
from skimage.exposure import match_histograms, adjust_log, adjust_gamma, adjust_sigmoid, equalize_adapthist, equalize_hist
from skimage.filters import unsharp_mask


nifti_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/nifti/'

info_df = pd.read_csv('data_info/info_official.csv')
foreground_df = pd.read_csv('data_info/foreground_info.csv')

# ======================================================================================
for pid, group, name in info_df[['pid', 'group', 'name']].values:
    images = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path + f'{group}-{name}.nii'))
    vmin, vmax = np.quantile(image, 0.001), np.quantile(image, 0.999)

    vmin = np.quantile(images, 0.001)
    vmax = np.quantile(images, 0.999)
    images = normalize(images, vmin=vmin, vmax=vmax)

    np.save(f'P:/REALTEK-HeadNeck-Project/OUS Mice/preprocess_data/P{pid:02d}.npy', images)


# ========================================================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
folds = []
for i, (_, test_index) in enumerate(skf.split(info_df, info_df['class'])):
    folds.append(test_index)


fold_info = np.zeros(len(info_df))
for i in range(5):
    fold_info[folds[i]] = i

info_df['fold'] = fold_info.astype(int)

# info_df.to_csv('data_info/info_split.csv', index=False)
# ========================================================================================
selected_foreground = foreground_df[(foreground_df.huang_foreground_ratio > 0.1) & (foreground_df.huang_foreground_normalized > 0.4)]
data_info = info_df.merge(selected_foreground[['pid', 'slice', 'huang_foreground_ratio', 'huang_foreground_normalized']], how='left', on='pid')
print(data_info.fold.value_counts())

# ========================================================================================

# index_order = np.arange(29)
# np.random.shuffle(index_order)

# images = [[] for _ in range(5)]
# pids = [[] for _ in range(5)]
# sids = [[] for _ in range(5)]
# targets = [[] for _ in range(5)]
# for idx in index_order:
#     info = info_df.iloc[idx]
#     pid = info['pid']
#     fold_idx = info['fold']
#     target = 0 if info['class'] == 'control' else 1
#     with open(f'P:/REALTEK-HeadNeck-Project/OUS Mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
#         img = np.load(f)
#     selected_slice = selected_foreground[selected_foreground.pid == pid]['slice']
#     # # get data
#     selected_img = img[list(selected_slice - 1)]
#     sids[fold_idx].extend(list(selected_slice))
#     images[fold_idx].extend(list(selected_img))
#     pids[fold_idx].extend([pid] * len(selected_slice))
#     targets[fold_idx].extend([target] * len(selected_slice))


# with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_2025.h5', 'w') as f:
#     for i in range(5):
#         f.create_group(f'fold_{i}')


# with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_2025.h5', 'a') as f:
#     for i in range(5):
#         img_data = np.array(images[i])
#         pid_data = np.array(pids[i])
#         sid_data = np.array(sids[i])
#         target_data = np.array(targets[i])
#         f[f'fold_{i}'].create_dataset(
#             'x', data=img_data[..., np.newaxis], dtype='f4')
#         f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
#         f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
#         f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')
# # ========================================================================================

with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_2025.h5', 'r') as f:
    for i in range(5):
        pids = f[f'fold_{i}']['patient_idx'][:]
        unique = [pids[0]]
        for pid in pids:
            if pid != unique[-1]:
                unique.append(pid)
        print(f'Fold_{i}', unique)
# Fold_0 [15, 8, 22, 17, 14, 29]
# Fold_1 [4, 9, 21, 5, 27, 24]
# Fold_2 [7, 1, 11, 18, 28, 19]
# Fold_3 [20, 13, 26, 25, 3, 6]
# Fold_4 [2, 16, 10, 23, 12]

pid_folds = [
    [15, 8, 22, 17, 14, 29],
    [4, 9, 21, 5, 27, 24],
    [7, 1, 11, 18, 28, 19],
    [20, 13, 26, 25, 3, 6],
    [2, 16, 10, 23, 12]
]
# ==========================================================================================
# All three channels

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = info_df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'P:/REALTEK-HeadNeck-Project/OUS Mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram equalization here
    img = np.stack([img, equalize_adapthist(
        img), unsharp_mask(img, radius=5, amount=2)], axis=-1)

    selected_slice = selected_foreground[selected_foreground.pid == pid]['slice']
    # # get selected_foreground
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_3c_2025.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_3c_2025.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_3c_2025.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('P:/REALTEK-HeadNeck-Project/OUS Mice/datasets/mice_2025.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)


# =====================================
# plot


# plt.subplot(1, 2, 1)
# plt.axis('off')
# with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
#     img = f['fold_0']['x'][5]
# plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
# plt.title('Q999')

# plt.subplot(1, 2, 2)
# plt.axis('off')
# with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'r') as f:
#     img = f['fold_0']['x'][5]
# plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
# plt.title('Adaptive hist')
# plt.show()


# with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
#     print(f['fold_0']['patient_idx'][5])
#     print(f['fold_0']['slice_idx'][5])

# # 29 & 6
# df[df.pid == 29]
# # 28   29     14     5  irradiated

# vmin, vmax = np.inf, -np.inf
# for i in range(1, 31):
#     img = sitk.ReadImage(fn.format(group=14, name=5, slice=i))
#     img_array = sitk.GetArrayFromImage(img)
#     new_vmin, new_vmax = img_array.min(), img_array.max()
#     vmin = min(vmin, new_vmin)
#     vmax = max(vmax, new_vmax)

# img = sitk.ReadImage(fn.format(group=14, name=5, slice=6))
# original_img = sitk.GetArrayFromImage(img)


# plt.subplot(2, 4, 1)
# plt.axis('off')
# plt.imshow(original_img[0], 'gray', vmin=vmin, vmax=vmax)
# plt.title('Original')

# with h5py.File('D:/OUS_mice/datasets/mice_3c.h5', 'r') as f:
#     img = f['fold_0']['x'][5]

# plt.subplot(2, 4, 2)
# plt.axis('off')
# plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
# plt.title('Quantile 0.001-0.999 (Q)')

# plt.subplot(2, 4, 3)
# plt.axis('off')
# plt.imshow(img[..., 1], 'gray', vmin=0, vmax=1)
# plt.title('Q + Adaptive Histogram Equalization')

# plt.subplot(2, 4, 4)
# plt.axis('off')
# plt.imshow(img[..., 2], 'gray', vmin=0, vmax=1)
# plt.title('Q + Unsharp filtered')

# with h5py.File('D:/OUS_mice/datasets/mice_MH_3c.h5', 'r') as f:
#     img = f['fold_0']['x'][5]
# plt.subplot(2, 4, 5)
# plt.axis('off')
# plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
# plt.title('Q + Matching histogram (MH)')

# plt.subplot(2, 4, 6)
# plt.axis('off')
# plt.imshow(img[..., 1], 'gray', vmin=0, vmax=1)
# plt.title('MH + Adaptive Histogram Equalization')

# plt.subplot(2, 4, 7)
# plt.axis('off')
# plt.imshow(img[..., 2], 'gray', vmin=0, vmax=1)
# plt.title('MH + Unsharp filtered')


# plt.tight_layout()
# plt.show()
