import os
import random
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

IMAGE_PATH = Path("data/images/")
LABEL_PAT = r'(.+)_\d+.jpg$'

with open("classes.txt", 'r') as f:
    CLASS_NAMES = list(map(lambda x: x.replace('\n', ''), f.readlines()))


def get_class(fname, pat=LABEL_PAT):
    return re.findall(pat, fname)[0]


def get_label(fname):

    class_name = get_class(fname)

    return CLASS_NAMES.index(class_name)


def is_corrupted(image_path):
    image = cv2.imread(str(image_path))

    return not isinstance(image, np.ndarray)


def find_corrupted_images(path_list):
    corrupted = []

    for image_path in path_list:
        if is_corrupted(image_path):
            corrupted.append(image_path)

    return corrupted


def filter_pathlist(path_list):
    corrupted_images = find_corrupted_images(path_list)

    for image_path in corrupted_images:
        path_list.remove(image_path)
        os.remove(str(image_path))
    print("Removed {} corrupted images.".format(len(corrupted_images)))

    return path_list


def split(path_list, valid_pct, shuffle=True):
    valid_len = int(valid_pct * len(path_list))
    total_freq = get_frequency(path_list, True)
    valid_freq = {label: int(valid_len * tf)
                  for label, tf in total_freq.items()}

    if shuffle:
        random.shuffle(path_list)

    valid_path_list = []
    train_path_list = path_list.copy()

    for path in path_list:
        label = get_class(path.name)

        if valid_freq[label] != 0:
            valid_path_list.append(path)
            valid_freq[label] -= 1
            train_path_list.remove(path)

        if list(valid_freq.values()) == 0:
            break

    return train_path_list, valid_path_list


def get_frequency(path_list, normalized=False):
    frequency = {}

    for path in path_list:
        key = get_class(path.name)

        if key in frequency:
            frequency[key] += 1
        else:
            frequency[key] = 1

    if normalized:
        frequency = {key: val / len(path_list)
                     for key, val in frequency.items()}

    return frequency


def plot_frequency(frequency):
    g = sns.barplot(x=list(frequency.keys()), y=list(frequency.values()))
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.show()
