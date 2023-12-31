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


path = 'D:/OUS_mice/Classes_from_experiments_DICOM/'
control_path = path + 'Control/'
irrad_path = path + 'Irradiated/'

df = pd.read_csv('data_info/info_official.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')

# ======================================================================================
for index, item in df.iterrows():
    if item['class'] == 'control':
        fn = control_path + 'Control_{group}_{name}_{slice}.dcm'
    else:
        fn = irrad_path + 'Irr_{group}_{name}_{slice}.dcm'
    images = []
    for i in range(30):
        img = sitk.ReadImage(
            fn.format(group=item['group'], name=item['name'], slice=i+1))
        images.append(sitk.GetArrayFromImage(img))
    images = np.concatenate(images, axis=0)
    vmin = images.min()
    vmax = images.max()
    print(vmin, vmax, images.mean(), images.std(), np.quantile(images, 0.9999))

    vmin = np.quantile(images, 0.001)
    vmax = np.quantile(images, 0.999)
    images = normalize(images, vmin=vmin, vmax=vmax)

    np.save(f'D:/OUS_mice/preprocess_data/P{item["pid"]:02d}.npy', images)


# ========================================================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
folds = []
for i, (_, test_index) in enumerate(skf.split(df, df['class'])):
    folds.append(test_index)


fold_info = np.zeros(len(df))
for i in range(5):
    fold_info[folds[i]] = i

df['fold'] = fold_info.astype(int)

# df.to_csv('data_info/info_split.csv', index=False)
# ========================================================================================

data_info = df.merge(df_selected[['pid', 'slice']], how='left', on='pid')
print(data_info.fold.value_counts())

# ========================================================================================

# index_order = np.arange(29)
# np.random.shuffle(index_order)

# images = [[] for _ in range(5)]
# pids = [[] for _ in range(5)]
# sids = [[] for _ in range(5)]
# targets = [[] for _ in range(5)]
# for idx in index_order:
#     info = df.iloc[idx]
#     pid = info['pid']
#     fold_idx = info['fold']
#     target = 0 if info['class'] == 'control' else 1
#     with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
#         img = np.load(f)
#     selected_slice = df_selected[df_selected.pid == pid]['slice']
#     # # get data
#     selected_img = img[list(selected_slice - 1)]
#     sids[fold_idx].extend(list(selected_slice))
#     images[fold_idx].extend(list(selected_img))
#     pids[fold_idx].extend([pid] * len(selected_slice))
#     targets[fold_idx].extend([target] * len(selected_slice))


# with h5py.File('D:/OUS_mice/datasets/mice.h5', 'w') as f:
#     for i in range(5):
#         f.create_group(f'fold_{i}')


# with h5py.File('D:/OUS_mice/datasets/mice.h5', 'a') as f:
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
# ========================================================================================

with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    for i in range(5):
        pids = f[f'fold_{i}']['patient_idx'][:]
        unique = [pids[0]]
        for pid in pids:
            if pid != unique[-1]:
                unique.append(pid)
        print(f'Fold_{i}', unique)
# Fold_0 [29, 14, 22, 17, 15, 8]
# Fold_1 [9, 27, 21, 4, 5, 24]
# Fold_2 [19, 1, 7, 18, 11, 28]
# Fold_3 [3, 6, 13, 25, 26, 20]
# Fold_4 [10, 12, 16, 2, 23]

pid_folds = [
    [29, 14, 22, 17, 15, 8],
    [9, 27, 21, 4, 5, 24],
    [19, 1, 7, 18, 11, 28],
    [3, 6, 13, 25, 26, 20],
    [10, 12, 16, 2, 23],
]
# ==========================================================================================

# adaptive histogram equalization
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram equalization here
    img = equalize_adapthist(img)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)

plt.subplot(1, 2, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 2, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Adaptive hist')
plt.show()


# =====================================================================

# unsharp filter
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram equalization here
    img = unsharp_mask(img, radius=5, amount=2)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_unsharp.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_unsharp.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('D:/OUS_mice/datasets/mice_unsharp.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)

plt.subplot(1, 3, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 3, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Adaptive hist')

plt.subplot(1, 3, 3)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_unsharp.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Unsharp')
plt.show()
# ===========================================================================
# All three channels

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram equalization here
    img = np.stack([img, equalize_adapthist(
        img), unsharp_mask(img, radius=5, amount=2)], axis=-1)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_3c.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_3c.h5', 'a') as f:
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
    with h5py.File('D:/OUS_mice/datasets/mice_3c.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)


# =====================================
# ==========================================================================================

# matching histogram
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')

with open(f'D:/OUS_mice/preprocess_data/P10.npy', 'rb') as f:
    ref_img = np.load(f)

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram matching here
    img = match_histograms(img, ref_img)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_MH.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_MH.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('D:/OUS_mice/datasets/mice_MH.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)

plt.subplot(1, 2, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 2, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_MH.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Matching hist')
plt.show()


# ===========================================================================
# Matching histogram then all three channels

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')

with open(f'D:/OUS_mice/preprocess_data/P10.npy', 'rb') as f:
    ref_img = np.load(f)


index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram matching here
    img = match_histograms(img, ref_img)
    # create 3 channels here
    img = np.stack([img, equalize_adapthist(
        img), unsharp_mask(img, radius=5, amount=2)], axis=-1)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_MH_3c.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_MH_3c.h5', 'a') as f:
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
    with h5py.File('D:/OUS_mice/datasets/mice_MH_3c.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)


# matching historgram + adaptive histogram equalization
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')
with open(f'D:/OUS_mice/preprocess_data/P10.npy', 'rb') as f:
    ref_img = np.load(f)


images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram matching here
    img = match_histograms(img, ref_img)
    # histogram equalization here
    img = equalize_adapthist(img)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_MH_AH.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_MH_AH.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('D:/OUS_mice/datasets/mice_MH_AH.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)

plt.subplot(1, 2, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 2, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_MH_AH.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('MH + Adaptive hist')
plt.show()


# matching historgram + unsharp
df = pd.read_csv('data_info/info_split.csv')
df_selected = pd.read_csv('data_info/info_selected.csv')
with open(f'D:/OUS_mice/preprocess_data/P10.npy', 'rb') as f:
    ref_img = np.load(f)


images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
index_order = np.concatenate(pid_folds) - 1
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    # histogram matching here
    img = match_histograms(img, ref_img)
    # histogram equalization here
    img = unsharp_mask(img, radius=5, amount=2)

    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice_MH_unsharp.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice_MH_unsharp.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset(
            'x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')


for i in range(5):
    with h5py.File('D:/OUS_mice/datasets/mice_MH_unsharp.h5', 'r') as f:
        patient_idx = f[f'fold_{i}']['patient_idx'][:]
        slice_idx = f[f'fold_{i}']['slice_idx'][:]
    with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
        slice_idx_original = f[f'fold_{i}']['slice_idx'][:]
        patient_idx_original = f[f'fold_{i}']['patient_idx'][:]
    assert np.all(slice_idx == slice_idx_original)

plt.subplot(1, 2, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 2, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_MH_unsharp.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('MH + Adaptive hist')
plt.show()


# =====================================
# plot


plt.subplot(1, 2, 1)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q999')

plt.subplot(1, 2, 2)
plt.axis('off')
with h5py.File('D:/OUS_mice/datasets/mice_AH.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Adaptive hist')
plt.show()


with h5py.File('D:/OUS_mice/datasets/mice.h5', 'r') as f:
    print(f['fold_0']['patient_idx'][5])
    print(f['fold_0']['slice_idx'][5])

# 29 & 6
df[df.pid == 29]
# 28   29     14     5  irradiated

vmin, vmax = np.inf, -np.inf
for i in range(1, 31):
    img = sitk.ReadImage(fn.format(group=14, name=5, slice=i))
    img_array = sitk.GetArrayFromImage(img)
    new_vmin, new_vmax = img_array.min(), img_array.max()
    vmin = min(vmin, new_vmin)
    vmax = max(vmax, new_vmax)

img = sitk.ReadImage(fn.format(group=14, name=5, slice=6))
original_img = sitk.GetArrayFromImage(img)


plt.subplot(2, 4, 1)
plt.axis('off')
plt.imshow(original_img[0], 'gray', vmin=vmin, vmax=vmax)
plt.title('Original')

with h5py.File('D:/OUS_mice/datasets/mice_3c.h5', 'r') as f:
    img = f['fold_0']['x'][5]

plt.subplot(2, 4, 2)
plt.axis('off')
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Quantile 0.001-0.999 (Q)')

plt.subplot(2, 4, 3)
plt.axis('off')
plt.imshow(img[..., 1], 'gray', vmin=0, vmax=1)
plt.title('Q + Adaptive Histogram Equalization')

plt.subplot(2, 4, 4)
plt.axis('off')
plt.imshow(img[..., 2], 'gray', vmin=0, vmax=1)
plt.title('Q + Unsharp filtered')

with h5py.File('D:/OUS_mice/datasets/mice_MH_3c.h5', 'r') as f:
    img = f['fold_0']['x'][5]
plt.subplot(2, 4, 5)
plt.axis('off')
plt.imshow(img[..., 0], 'gray', vmin=0, vmax=1)
plt.title('Q + Matching histogram (MH)')

plt.subplot(2, 4, 6)
plt.axis('off')
plt.imshow(img[..., 1], 'gray', vmin=0, vmax=1)
plt.title('MH + Adaptive Histogram Equalization')

plt.subplot(2, 4, 7)
plt.axis('off')
plt.imshow(img[..., 2], 'gray', vmin=0, vmax=1)
plt.title('MH + Unsharp filtered')


plt.tight_layout()
plt.show()
