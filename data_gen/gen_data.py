import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from deoxys_image import normalize
import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold
import h5py


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

df.to_csv('data_info/info_split.csv', index=False)
# ========================================================================================

data_info = df.merge(df_selected[['pid', 'slice']], how='left', on='pid')
print(data_info.fold.value_counts())

# ========================================================================================

index_order = np.arange(29)
np.random.shuffle(index_order)

images = [[] for _ in range(5)]
pids = [[] for _ in range(5)]
sids = [[] for _ in range(5)]
targets = [[] for _ in range(5)]
for idx in index_order:
    info = df.iloc[idx]
    pid = info['pid']
    fold_idx = info['fold']
    target = 0 if info['class'] == 'control' else 1
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        img = np.load(f)
    selected_slice = df_selected[df_selected.pid == pid]['slice']
    # # get data
    selected_img = img[list(selected_slice - 1)]
    sids[fold_idx].extend(list(selected_slice))
    images[fold_idx].extend(list(selected_img))
    pids[fold_idx].extend([pid] * len(selected_slice))
    targets[fold_idx].extend([target] * len(selected_slice))


with h5py.File('D:/OUS_mice/datasets/mice.h5', 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')


with h5py.File('D:/OUS_mice/datasets/mice.h5', 'a') as f:
    for i in range(5):
        img_data = np.array(images[i])
        pid_data = np.array(pids[i])
        sid_data = np.array(sids[i])
        target_data = np.array(targets[i])
        f[f'fold_{i}'].create_dataset('x', data=img_data[..., np.newaxis], dtype='f4')
        f[f'fold_{i}'].create_dataset('y', data=target_data, dtype='f4')
        f[f'fold_{i}'].create_dataset('patient_idx', data=pid_data, dtype='i4')
        f[f'fold_{i}'].create_dataset('slice_idx', data=sid_data, dtype='i4')
