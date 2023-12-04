import os
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms, adjust_log, adjust_gamma, adjust_sigmoid, equalize_adapthist, equalize_hist
from skimage.filters import unsharp_mask

# Adjust log (log correction) --> does not change much
for pid in range(1, 30):
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        images = np.load(f)

    images = adjust_log(images)
    vmin, vmax = images.min(), images.max()

    plt.figure(figsize=(17, 10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'Pid {pid:02d}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_adjust_log/P{pid:02d}.png')
    plt.close('all')



# unsharp # nice one
for pid in range(1, 30):
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        images = np.load(f)

    images = unsharp_mask(images, radius=5, amount=2)
    vmin, vmax = images.min(), images.max()

    plt.figure(figsize=(17, 10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'Pid {pid:02d}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_unsharp/P{pid:02d}.png')
    plt.close('all')



# Adjust sigmoid # something to try (as another channel), gain=10 or gain=5
for pid in range(1, 30):
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        images = np.load(f)

    images = adjust_sigmoid(images, gain=5)
    vmin, vmax = images.min(), images.max()

    plt.figure(figsize=(17, 10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'Pid {pid:02d}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_adjust_sigmoid/P{pid:02d}.png')
    plt.close('all')


# better contrast
for pid in range(1, 30):
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        images = np.load(f)

    images = equalize_adapthist(images)
    vmin, vmax = images.min(), images.max()

    plt.figure(figsize=(17, 10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'Pid {pid:02d}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_adapthist/P{pid:02d}.png')
    plt.close('all')


# matching histogram # not really consistent
# pid = 10 (control 8-7)
with open(f'D:/OUS_mice/preprocess_data/P10.npy', 'rb') as f:
    ref_img = np.load(f)
for pid in range(1, 30):
    with open(f'D:/OUS_mice/preprocess_data/P{pid:02d}.npy', 'rb') as f:
        images = np.load(f)

    images = match_histograms(images, ref_img)
    vmin, vmax = images.min(), images.max()

    plt.figure(figsize=(17, 10))
    for i in range(30):
        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(images[i], 'gray', vmin=vmin, vmax=vmax)
        # plt.imshow(images[i][:, first:last+1] > best_threshold, 'gray')
        # plt.imshow(images[i][:, first: last+1], 'gray')
        plt.title(i+1)
    plt.suptitle(f'Pid {pid:02d}')
    plt.tight_layout()
    plt.savefig(f'D:/OUS_mice/visualize_MH/P{pid:02d}.png')
    plt.close('all')
