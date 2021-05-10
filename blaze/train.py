# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import os
from pathlib import Path

import torch
import torch.optim as optim
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import Frames, PaintedFramesDataLoader
from .losses import full_loss
from .model import BlazeNet, anchor_boxes
from .polygons import blend
from .utils import targets_to_batch

# %%
runs_dir = Path("/home/malpunek/coding/droniada/runs")
os.makedirs(runs_dir, exist_ok=True)

run_num = len(list(runs_dir.iterdir()))
run_dir = runs_dir / f"{run_num:03d}"

while run_dir.exists():
    run_num += 1
    run_dir = runs_dir / f"{run_num:03d}"

os.makedirs(run_dir)
writer = SummaryWriter(run_dir)

# %%
frames = Frames("/data/droniada/back256_144.mp4")

writer.add_image("Random background image", tv.transforms.ToTensor()(frames.random()))

# %%
img, figs = blend(frames.random(), frames.random())
writer.add_image_with_boxes(
    "Sample",
    tv.transforms.ToTensor()(img),
    tv.ops.box_convert(figs[..., :4], "cxcywh", "xyxy"),
)

writer.add_text("Figures", str(figs))

# %%
torch.autograd.set_detect_anomaly(True)

# %%


def get_eval_sample():
    data = Frames("/data/droniada/eval256_144.mp4")
    trans = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,), (0.5,))]
    )
    eframes = [trans(data.random()) for _ in range(16)]
    return torch.stack(eframes)


def plot_eval_sample(writer, model, sample, global_step, topk=None):
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
        if topk is None:
            my_boxes = my_boxes[my_boxes[..., 0] > 0.5]
        else:
            _, idxs = torch.topk(my_boxes[..., 0].reshape(-1), topk)
            my_boxes = my_boxes.reshape(-1, 5)[idxs]

        return tv.ops.box_convert(my_boxes[..., 1:], "cxcywh", "xyxy")

    outs = torch.stack(
        [tv.utils.draw_bounding_boxes(img, get_boxes(i)) for i, img in enumerate(imgs)]
    )

    grid = tv.utils.make_grid(outs)
    name = "Eval Grid" if topk is None else f"Eval topk {topk}"
    writer.add_image(name, grid, global_step=global_step)


eval_sample = get_eval_sample()

# %%
epoch_size = 3
batch_size = 32
num_epochs = 2

loader = PaintedFramesDataLoader(frames, batch_size, epoch_size)

model = BlazeNet(4, 4)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


model.apply(weights_init)
optimizer = optim.Adam(model.parameters())

device = torch.device("cuda")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for local_step, (inputs, targets) in enumerate(tqdm(loader, total=epoch_size)):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        anchors = anchor_boxes(tuple(outputs.shape[1:3]))

        formatted_targets = targets_to_batch(targets, outputs, anchors).to(device)
        loss, [obj_loss, cls_loss, cent_loss, size_loss] = full_loss(
            outputs, formatted_targets
        )

        loss.backward()
        optimizer.step()

        step = epoch * epoch_size + local_step
        writer.add_scalar("Objectness loss", obj_loss, global_step=step)
        writer.add_scalar("Center loss", cent_loss, global_step=step)
        writer.add_scalar("Sizes loss", size_loss, global_step=step)
        writer.add_scalar("Classification loss", cls_loss, global_step=step)
        writer.add_scalar("Full loss", loss, global_step=step)

        running_loss += loss.item() / batch_size
    writer.add_scalar("Epoch loss", loss, global_step=(epoch + 1) * epoch_size)
    plot_eval_sample(writer, model, eval_sample, (epoch + 1) * epoch_size)
    plot_eval_sample(writer, model, eval_sample, (epoch + 1) * epoch_size, topk=15)

    torch.save(model.state_dict(), run_dir / f"state_dict_e{epoch:03d}.pth")


# %%
