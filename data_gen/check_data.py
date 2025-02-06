import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, median_absolute_error, max_error

path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/OUS_mice/Classes_from_experiments_DICOM/'
control_path = path + 'Control'
irrad_path = path + 'Irradiated'


os.listdir(control_path)

control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)

irrad_df


control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'

pd.concat([control_df, irrad_df]).reset_index(drop=True).to_csv(
    'data_info/info_tmp.csv', index_label='pid')


control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).groupby(['group', 'name']).count().reset_index()


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).groupby(['group', 'name']).count().reset_index()
np.unique(control_df['group'].astype(str) + '_' +
          control_df['name'].astype(str).values)
np.unique(irrad_df['group'].astype(str) + '_' +
          irrad_df['name'].astype(str).values)


control_df
irrad_df

control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'


df = pd.concat([control_df[['group', 'name', 'class']], irrad_df[['group', 'name', 'class']]]
               ).reset_index(drop=True)
df.index = df.index + 1

df.to_csv('data_info/info_official.csv', index_label='pid')

# ===========================================================================================================================
path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/OUS_mice/Visually_Filtered_300_per_class_Jpeg/'
control_path = path + 'control'
irrad_path = path + 'irradiated'

control_df = pd.DataFrame(np.array([fn[8: -4].split('_') for fn in os.listdir(control_path)]).astype(int),
                          columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


irrad_df = pd.DataFrame(np.array([fn[4: -4].split('_') for fn in os.listdir(irrad_path)]).astype(int),
                        columns=['group', 'name', 'slice']).sort_values(['group', 'name', 'slice']).reset_index(drop=True)


control_df['class'] = 'control'
irrad_df['class'] = 'irradiated'
pd.concat([control_df, irrad_df])

pd.read_csv('data_info/info_official.csv').merge(
    pd.concat([control_df, irrad_df]), how='left', on=['group', 'name', 'class']
).to_csv('data_info/info_selected.csv', index=False)

# ===========================================================================================================================
# check dicom series


def find_dicom_series(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image, [fn.split('/')[-1] for fn in dicom_names]


def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    binary_image = otsu_filter.Execute(image)
    return sitk.GetArrayFromImage(binary_image)


def apply_huang_threshold(image):
    # https://www.researchgate.net/publication/236207740_Single_Cell_Analysis_of_Drug_Distribution_by_Intravital_Imaging
    # Apply Huang's thresholding
    huang_filter = sitk.HuangThresholdImageFilter()
    huang_filter.SetInsideValue(0)
    huang_filter.SetOutsideValue(1)
    binary_image = huang_filter.Execute(image)
    return sitk.GetArrayFromImage(binary_image)


# Example usage
path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/raw_data/'
control_path = path + 'control'
irrad_path = path + 'irradiated'
otsu_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/otsu_threshold/'
huang_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/huang_threshold/'

# Process control series
for patient in os.listdir(control_path):
    image, dicom_names = find_dicom_series(control_path + '/' + patient)
    binary_image = apply_otsu_threshold(image)
    with open(otsu_path + '/images/' + patient + '.npy', 'wb') as f:
        np.save(f, binary_image)
    if not os.path.exists(otsu_path + '/vis/' + patient):
        os.mkdir(otsu_path + '/vis/' + patient)
    for i, name in enumerate(dicom_names):
        plt.imshow(binary_image[i], cmap='gray')
        plt.axis('off')
        plt.savefig(otsu_path + '/vis/' + patient + f'/{name[:-4]}.png')
        plt.close('all')

# Process irradiated series
for patient in os.listdir(irrad_path):
    image, dicom_names = find_dicom_series(irrad_path + '/' + patient)
    binary_image = apply_otsu_threshold(image)
    with open(otsu_path + '/images/' + patient + '.npy', 'wb') as f:
        np.save(f, binary_image)
    if not os.path.exists(otsu_path + '/vis/' + patient):
        os.mkdir(otsu_path + '/vis/' + patient)
    for i, name in enumerate(dicom_names):
        plt.imshow(binary_image[i], cmap='gray')
        plt.axis('off')
        plt.savefig(otsu_path + '/vis/' + patient + f'/{name[:-4]}.png')
        plt.close('all')


for path in [control_path, irrad_path]:
    for patient in os.listdir(path):
        image, dicom_names = find_dicom_series(path + '/' + patient)
        binary_image = apply_huang_threshold(image)
        with open(huang_path + '/images/' + patient + '.npy', 'wb') as f:
            np.save(f, binary_image)
        if not os.path.exists(huang_path + '/vis/' + patient):
            os.mkdir(huang_path + '/vis/' + patient)
        for i, name in enumerate(dicom_names):
            plt.imshow(binary_image[i], cmap='gray')
            plt.axis('off')
            plt.savefig(huang_path + '/vis/' + patient + f'/{name[:-4]}.png')
            plt.close('all')


nifti_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/nifti/'
for path in [control_path, irrad_path]:
    for patient in os.listdir(path):
        image, dicom_names = find_dicom_series(path + '/' + patient)
        sitk.WriteImage(image, nifti_path + patient + '.nii')


# ===========================================================================================================================
# compare selections
info = pd.read_csv('data_info/info_official.csv')
selected = pd.read_csv('data_info/info_selected.csv')

count_df = selected.groupby(
    ['pid', 'group', 'name']).size().reset_index(name='manish_count')

# Display the result
info = info.merge(count_df[['pid', 'manish_count']], how='left', on='pid')

data = {'pid': [], 'group': [], 'name': [], 'slice': [],
        'otsu_foreground': [], 'huang_foreground': []}
for pid, group, name in info[['pid', 'group', 'name']].values:
    with open(otsu_path + f'/images/{group}-{name}.npy', 'rb') as f:
        otsu_image = np.load(f)
    with open(huang_path + f'/images/{group}-{name}.npy', 'rb') as f:
        huang_image = np.load(f)
    foreground_otsu = otsu_image.sum(axis=(1, 2))
    foreground_huang = huang_image.sum(axis=(1, 2))
    data['pid'].extend([pid] * len(foreground_otsu))
    data['group'].extend([group] * len(foreground_otsu))
    data['name'].extend([name] * len(foreground_otsu))
    data['slice'].extend(np.arange(1, len(foreground_otsu) + 1))
    data['otsu_foreground'].extend(foreground_otsu)
    data['huang_foreground'].extend(foreground_huang)


foreground_df = pd.DataFrame(data)
foreground_df['otsu_foreground_ratio'] = foreground_df['otsu_foreground'] / 256 / 256
foreground_df['huang_foreground_ratio'] = foreground_df['huang_foreground'] / 256 / 256
foreground_df[foreground_df['pid'] == 1]

# Group by 'pid' and 'name' and calculate the maximum 'otsu_foreground' for each group
max_otsu_foreground = foreground_df.groupby(['pid', 'group', 'name'])[
    'otsu_foreground'].transform('max')
max_huang_foreground = foreground_df.groupby(['pid', 'group', 'name'])[
    'huang_foreground'].transform('max')

# Divide 'otsu_foreground' by the maximum value within their respective groups
foreground_df['otsu_foreground_normalized'] = foreground_df['otsu_foreground'] / \
    max_otsu_foreground
foreground_df['huang_foreground_normalized'] = foreground_df['huang_foreground'] / \
    max_huang_foreground
# Display the updated DataFrame
print(foreground_df[foreground_df['pid'] == 1])

foreground_df.to_csv('data_info/foreground_info.csv', index=False)

# Define the thresholds
otsu_threshold = 0.05
huang_threshold = 0.1
otsu_normalized_threshold = 0.36
huang_normalized_threshold = 0.4

thres_info = {'otsu_foreground_ratio': otsu_threshold,
              'huang_foreground_ratio': huang_threshold,
              'otsu_foreground_normalized': otsu_normalized_threshold,
              'huang_foreground_normalized': huang_normalized_threshold}

# count df
info_extended = info.copy()
for name, thres in thres_info.items():
    count_df = foreground_df.groupby(['pid', 'group', 'name']).apply(
        lambda x: (x[name] > thres).sum()).reset_index(name=name + '_count')
    info_extended = info_extended.merge(count_df, on=['pid', 'group', 'name'])
# union of the huang methods
count_df = foreground_df.groupby(['pid', 'group', 'name']).apply(
    lambda x: (((x['huang_foreground_ratio'] > huang_threshold).astype(float) + (x['huang_foreground_normalized'] > huang_normalized_threshold).astype(float)) > 0).sum()).reset_index(name='huang_union_count')
info_extended = info_extended.merge(count_df, on=['pid', 'group', 'name'])
# intersection of the huang methods
count_df = foreground_df.groupby(['pid', 'group', 'name']).apply(
    lambda x: (((x['huang_foreground_ratio'] > huang_threshold).astype(float) * (x['huang_foreground_normalized'] > huang_normalized_threshold).astype(float)) > 0).sum()).reset_index(name='huang_inter_count')
info_extended = info_extended.merge(count_df, on=['pid', 'group', 'name'])
info_extended.to_csv('data_info/info_extended.csv', index=False)

for key in thres_info.keys() | {'huang_union', 'huang_inter'}:
    print(key, mean_squared_error(info_extended['manish_count'], info_extended[key + '_count']),
          median_absolute_error(info_extended['manish_count'], info_extended[key + '_count']),
          max_error(info_extended['manish_count'], info_extended[key + '_count']))

info_extended.sum()

# ===========================================================================================================================
# visualize thresholding results
info = pd.read_csv('data_info/info_extended.csv')
foreground_df = pd.read_csv('data_info/foreground_info.csv')
nifti_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/nifti/'
output_path = 'P:/REALTEK-HeadNeck-Project/OUS Mice/vis_threshold/'

for pid, group, name in info[['pid', 'group', 'name']].values:
    image = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path + f'{group}-{name}.nii'))
    vmin, vmax = np.quantile(image, 0.001), np.quantile(image, 0.999)
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Patient ID: {pid}, File: {group}-{name}')
    for i in range(image.shape[0]):
        data = foreground_df.filter(['pid', 'group', 'name',
                              'slice','huang_foreground_ratio',
                              'huang_foreground_normalized']).query(
            f'pid == {pid} and group == {group} and name == {name} and slice == {i + 1}')
        plt.subplot(6, 5, i + 1)
        plt.imshow(image[i], cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        extend = ''
        if data['huang_foreground_ratio'].values[0] > huang_threshold:
            extend += ' > R'
        if data['huang_foreground_normalized'].values[0] > huang_normalized_threshold:
            extend += ' > N'
        plt.title(f'Slice {i}' + extend)
    plt.savefig(output_path + f'{group}-{name}.png')
    plt.close('all')
