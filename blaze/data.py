import random as pyrand

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Frames(Dataset):
    def __init__(self, filepath, transform=None):
        self.reader = cv2.VideoCapture(filepath)
        self.transform = transform

    def __len__(self):
        return int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        if idx > len(self):
            raise KeyError
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.reader.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def random(self):
        return self[pyrand.randint(0, len(self))]


def to_normal_coords(targets, imgshape):
    """Converts absolute pixel coordinates to ones normalized through image size
    Args:
        targets: torch.Tensor(objects x [x, y...])
        imgshapes: torch.Tensor([channel, H, W])
    """
    targets[..., 0::2] = targets[..., 0::2] / imgshape[2]
    targets[..., 1::2] = targets[..., 1::2] / imgshape[1]
    return targets


def to_blaze_class(classes, num_classes=3):
    """
    [0, 0, 1, 2] => tensor([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    """
    F.one_hot(classes, num_classes=num_classes)
