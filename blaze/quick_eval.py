# %%

try:
    __IPYTHON__
    import sys

    sys.path.insert(0, "/home/malpunek/coding/droniada")
    __package__ = "blaze"

    # sys.argv = ["nothing.py", "-t", "25", "-x", "999"]

except NameError:
    pass

# %%
import torch
import torchvision as tv
from matplotlib.pyplot import imshow

from .model import BlazeNet
from .utils import get_eval_sample

device = torch.device("cpu")
model = BlazeNet(4, 4)
model.load_state_dict(
    torch.load(
        "/home/malpunek/coding/droniada/runs/001/state_dict_e299.pth",
        map_location=device,
    )
)


def get_boxes(model, sample, iou_threshold=0.5):
    device = next(model.parameters()).device
    sample = sample.to(device)
    outputs = model(sample).cpu()
    sample = sample.cpu()

    imgs = [
        ((sample[i] + 1) * 127).to(dtype=torch.uint8) for i in range(sample.size(0))
    ]

    boxes = outputs[..., :5]
    boxes[..., 1::2] *= sample.size(3)
    boxes[..., 2::2] *= sample.size(2)

    def get_boxes(i):
        my_boxes = boxes[i]
        my_boxes = my_boxes[my_boxes[..., 0] > iou_threshold]
        indices = tv.ops.batched_nms(
            my_boxes[..., 1:], my_boxes[..., 0], torch.zeros_like(my_boxes[..., 0]), 0.1
        )

        # print(my_boxes[..., 1:].shape)
        return tv.ops.box_convert(my_boxes[indices][..., 1:], "cxcywh", "xyxy")

    outs = torch.stack(
        [tv.utils.draw_bounding_boxes(img, get_boxes(i)) for i, img in enumerate(imgs)]
    )

    grid = tv.utils.make_grid(outs)
    return grid


# %%
from matplotlib.pyplot import figure

figure(figsize=(20, 6), dpi=80)

sample = get_eval_sample()
imshow(get_boxes(model, sample, 0.9).permute(1, 2, 0))

# %%
def calc_boxes(model, sample, iou_threshold=0.5):
    outputs = model(sample)

    boxes = outputs[..., :5]
    boxes[..., 1::2] *= sample.size(3)
    boxes[..., 2::2] *= sample.size(2)
    return boxes

# %%
