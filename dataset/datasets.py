import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import os
from collections import defaultdict


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples, label_to_indexes = self._make_dataset(self.root_dir, class_to_idx)

        print('samples num', len(samples))
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.label_to_indexes = label_to_indexes
        self.classes = sorted(self.label_to_indexes.keys())
        print('class num', len(self.classes))


    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


    def _make_dataset(self, root_dir, class_to_idx):
        root_dir = os.path.expanduser(root_dir)
        images = []
        label2index = defaultdict(list)
        image_index = 0
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith(('jpg','png')):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target]) 
                        images.append(item)
                        label2index[class_to_idx[target]].append(image_index) 
                        image_index += 1
        return images, label2index

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        
        if (sample.mode != 'RGB'):
            sample = sample.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
