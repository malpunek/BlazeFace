from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeConv(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1):
        super().__init__()

        self.channels, self.stride = channels, stride

        # self.padding = (kernel_size - 1) // 2 if stride == 1 else math.ceil(kernel_size / 2) - 1
        self.padding = (kernel_size - 1) // 2

        self.main = nn.Sequential(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size,
                stride=stride,
                groups=channels[0],
                padding=self.padding,
            ),
            nn.BatchNorm2d(channels[1]),
            nn.Conv2d(channels[1], channels[2], 1),
            nn.BatchNorm2d(channels[2]),
        )

    def forward(self, x):
        res = self.main(x)
        return res


class BlazeSkip(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()

        self.channels, self.stride = channels, stride

    def forward(self, x):
        if self.stride == 2:
            pad_h, pad_w = x.shape[2] % 2, x.shape[3] % 2
            if pad_h + pad_w != 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            x = F.max_pool2d(x, self.stride)
        if self.channels[-1] > self.channels[0]:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channels[-1] - self.channels[0]))
        return x


class SingleBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1):
        super().__init__()

        self.main = BlazeConv(channels, kernel_size=kernel_size, stride=stride)
        self.residual = BlazeSkip(channels, stride=stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        m = self.main(x)
        r = self.residual(x)
        return self.activation(m + r)


class DoubleBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, stride=1):
        super().__init__()

        self.main = nn.Sequential(
            BlazeConv(
                channels[:3],
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(inplace=True),
            BlazeConv(channels[-3:], kernel_size=kernel_size, stride=1),
        )

        self.residual = BlazeSkip(channels, stride=stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        m = self.main(x)
        r = self.residual(x)
        return self.activation(m + r)


class BlazeFeats(nn.Module):
    def __init__(self):
        super().__init__()

        self.features_16 = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            SingleBlazeBlock([24, 24, 24]),
            SingleBlazeBlock([24, 24, 24]),
            SingleBlazeBlock([24, 24, 48], stride=2),
            SingleBlazeBlock([48, 48, 48]),
            SingleBlazeBlock([48, 48, 48]),
            DoubleBlazeBlock([48, 48, 24, 24, 96], stride=2),
            DoubleBlazeBlock([96, 96, 24, 24, 96]),
            DoubleBlazeBlock([96, 96, 24, 24, 96]),
        )

        self.features_8 = nn.Sequential(
            DoubleBlazeBlock([96, 96, 24, 24, 96], stride=2),
            DoubleBlazeBlock([96, 96, 24, 24, 96]),
            DoubleBlazeBlock([96, 96, 24, 24, 96]),
        )

    def forward(self, x):
        """
        Args:
            x: (N, 3, 144, 256) || (N, 3, 128, 128)
        """
        f16 = self.features_16(x)  # (N, 96, 18, 32)  || (N, 96, 16, 16)
        f8 = self.features_8(f16)  # (N, 96, 9, 16) || (N, 96, 8, 8)
        return f16, f8


class BlazePred(nn.Module):
    def __init__(self, in_channels, num_classes, anchors_per_cell):
        super().__init__()
        self.num_classes = num_classes
        self.predictions = nn.Conv2d(
            in_channels, (5 + num_classes) * anchors_per_cell, 1
        )
        self.anchors_per_cell = anchors_per_cell

    def forward(self, x):
        """
        (N, in_channels, H, W) => (N, H, W, anchors, out)
        where out = [hasObject, x, y, w, h, class1_prob, class2_prob...]"""
        scores = self.predictions(x)  # (N, out * anchors, H, W)
        scores = scores.permute(0, 2, 3, 1)  # (N, H, W, out * anchors)
        scores = scores.reshape(
            *scores.shape[:3], self.anchors_per_cell, 5 + self.num_classes
        )  # (N, H, W, anchors, out)
        return scores


def to_concrete_scores(scores, anchors):
    """(N, H, W, anchors, out) => (N, H, W, anchors, out)"""
    cell_w, cell_h = 1 / scores.size(2), 1 / scores.size(1)
    scores[..., 1:3] = torch.tanh(scores[..., 1:3])
    scores[..., 1] = scores[..., 1] * (cell_w / 2) + anchors[..., 0]
    scores[..., 2] = scores[..., 2] * (cell_h / 2) + anchors[..., 1]
    scores[..., 3:5] = torch.exp(scores[..., 3:5])
    scores[..., 3:5] *= anchors[..., 2:4]
    result = scores.clone()
    result[..., 5:] = F.softmax(scores[..., 5:], dim=-1)
    return result


def anchor_centers(shape: tuple[int, int]):
    h, w = shape
    with torch.no_grad():
        xs = (torch.arange(w) * 2 + 1) / (2 * w)
        ys = (torch.arange(h) * 2 + 1) / (2 * h)

        centers = torch.cartesian_prod(ys, xs)
        centers = torch.index_select(centers, 1, torch.LongTensor([1, 0]))
        centers = centers.reshape(h, w, 2)

    return centers


@cache
def anchor_boxes(shape: tuple[int, int]):
    """
    Args:
        shape: (H, W)
    Returns:
        torch.Tensor(H, W, anchors, 4) where last dimension is (center_x, center_y, w, h)
    """
    h, w = shape
    with torch.no_grad():
        # TODO Config for sizes
        size_x = torch.Tensor([0.2, 0.5, 1, 2.3]) / w
        size_y = torch.Tensor([0.2, 0.5, 1, 2.3]) / h

        centers = anchor_centers(shape)
        sizes = torch.stack([size_x, size_y]).T

        result = torch.zeros(h, w, 1, 4)
        result[:, :, :, :2] = centers.reshape(h, w, 1, 2)
        result = result.expand(h, w, size_x.size(0), 4).clone()
        result[..., 2:] = sizes
        return result


class BlazeNet(nn.Module):
    def __init__(self, num_anchors_8, num_anchors_16):
        super().__init__()

        self.features = BlazeFeats()

        # self.predictions_16 = BlazePred(96, 3, num_anchors_16)
        self.predictions_8 = BlazePred(96, 3, num_anchors_8)

    def forward(self, x):
        f16, f8 = self.features(x)
        # pred16 = self.predictions_16(f16)
        pred8 = self.predictions_8(f8)
        device = next(self.parameters()).device
        anchors_8 = anchor_boxes(tuple(pred8.shape[1:3])).to(device)
        # anchors_16 = anchor_boxes(tuple(pred16.shape[1:3]))
        return to_concrete_scores(
            pred8, anchors_8
        )  # , to_concrete_scores(pred16, anchors_16)
