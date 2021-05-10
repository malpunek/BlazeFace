import random as pyrand

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .polygons import blend


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


class PaintedFramesDataLoader:
    def __init__(self, dataset, batch_size, num_batches, imgs_read=8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.imgs_read = imgs_read

    def __getitem__(self, i):
        if i >= self.num_batches:
            raise IndexError
        frames = [self.dataset.random() for i in range(self.imgs_read)]
        idxs = torch.randint(self.imgs_read, (self.batch_size, 2))
        blends = list(zip(*[blend(frames[i], frames[j]) for i, j in idxs]))
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        tensorFrames = [trans(img) for img in blends[0]]
        return torch.stack(tensorFrames), [normalize_bboxes(bs, tensorFrames[0].shape[1:]) for bs in blends[1]]


def to_normal_coords(targets, imgshape):
    """Converts absolute pixel coordinates to ones normalized through image size
    Args:
        targets: torch.Tensor(objects x [x, y...])
        imgshapes: torch.Tensor([channel, H, W])
    """
    targets[..., 0::2] = targets[..., 0::2] / imgshape[2]
    targets[..., 1::2] = targets[..., 1::2] / imgshape[1]
    return targets


def normalize_bboxes(bboxes, shape):
    h, w = shape
    bboxes[:, 0:4:2] /= w
    bboxes[:, 1:4:2] /= h
    return bboxes


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
