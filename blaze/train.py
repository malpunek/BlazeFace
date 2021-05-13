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
from .utils import get_eval_sample, plot_eval_sample, targets_to_batch

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
eval_sample = get_eval_sample()

# %%
torch.autograd.set_detect_anomaly(True)

# %%
epoch_size = 32
batch_size = 32
num_epochs = 300

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

for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
    running_loss = 0.0
    for local_step, (inputs, targets) in enumerate(
        tqdm(loader, total=epoch_size, desc="Iteration", leave=False)
    ):
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

    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), run_dir / f"state_dict_e{epoch:03d}.pth")


# %%
