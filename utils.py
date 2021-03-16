import os
import re

IMAGE_PATH = "data/images/"
LABEL_PAT = r'(.+)_\d+.jpg$'


def get_class(fname, pat=LABEL_PAT):
    return re.findall(pat, fname)[0]


def get_classes(path):
    fnames = os.listdir(path)
    all_classes = set([get_class(f) for f in fnames if ".jpg" in f])

    return list(all_classes)


CLASS_NAMES = get_classes(IMAGE_PATH)


def get_label(fname):
    
    class_name = get_class(fname)
    return CLASS_NAMES.index(class_name)
