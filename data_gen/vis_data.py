import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import SimpleITK as sitk


path = 'D:/OUS_mice/Classes_from_experiments_DICOM/'
control_path = path + 'Control/'
irrad_path = path + 'Irradiated/'

df = pd.read_csv('data_info/info_official.csv')

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
    print(vmin, vmax, images.mean(), images.std())
    # vmin = images.mean() - images.std()
    # vmax = images.mean() + 3*images.std()

    threshold_range = range(int(np.max(images)) + 1)
    criterias = [compute_otsu_criteria(images, thres) for thres in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]

    # first, last = np.argwhere((images > best_threshold).sum(axis=(0, 1)) > 10).flatten()[[0, -1]]
    # cropped = images[..., first:last+1]
    # print(vmin, vmax, cropped.mean(), cropped.std())
    # vmin = max(0, cropped.mean() - cropped.std())
    # vmax = cropped.mean() + 3*cropped.std()

    selected = images[images>best_threshold]
    print(vmin, vmax, selected.mean(), selected.std())
    vmin = images.min() #selected.mean() - selected.std()
    vmax = selected.mean() + selected.std()

    plt.figure(figsize=(17,10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'{item["class"]}_{item["group"]}_{item["name"]}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_tmp/{item["class"]}_{item["group"]}_{item["name"]}.png')
    plt.close('all')



(images > best_threshold).sum(axis=(0, 1))



plt.hist(images.flatten(), bins=50)
plt.show()


images[0,0,8]

hist = images.flatten().astype(int)

def compute_otsu_criteria(im, thres):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im>=thres] = 1

    nb_pixels = im.size
    nb_pixels_1 = thresholded_im.sum()
    weight1 = nb_pixels_1 / nb_pixels
    weight0 = 1 - weight1

    if weight1 == 0 or weight0 == 0:
        return np.inf

    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1




images[images > 23].mean()
images[images > 23].std()


images[..., first:last].mean()
images[..., first:last].std()


plt.hist(images[..., first:last].flatten(), bins=50)
plt.show()

images[images > best_threshold].mean()
images[images > best_threshold].std()

cropped

(images > best_threshold).sum(axis=(0, 1))

np.argwhere((images > best_threshold).sum(axis=(0, 1)))

(images>np.median(images)).sum()
images.size


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

    plt.figure(figsize=(17,10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'{item["class"]}_{item["group"]}_{item["name"]}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_q999/{item["class"]}_{item["group"]}_{item["name"]}.png')
    plt.close('all')



np.quantile(images, 0.999)
images[images > 221].size
10*20*10
images.shape

5*10*5


np.quantile(images, 0.0001)
