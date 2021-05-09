# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.optim as optim
from matplotlib.pyplot import imshow
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Frames, PaintedFramesDataLoader
from .losses import (classification_loss, full_loss, localization_loss,
                     objectness_loss)
from .model import BlazeNet, anchor_boxes
from .polygons import blend, preprocess
from .utils import targets_to_batch

# %%
frames = Frames("/data/droniada/back256_144.mp4")
img = frames.random()
# imshow(img)


# %%
img, figs = blend(frames.random(), frames.random())
print(figs)
# imshow(img)
torch.autograd.set_detect_anomaly(True)

# %%
epoch_size = 50
batch_size = 32
num_epochs = 20
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
    for inputs, targets in tqdm(loader, total=epoch_size):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs)
        anchors = anchor_boxes(tuple(outputs.shape[1:3]))

        formatted_targets = targets_to_batch(targets, outputs, anchors).to(device)
        loss = full_loss(outputs, formatted_targets)

        loss.backward()
        optimizer.step()

        losses = [
            objectness_loss(outputs, formatted_targets, 1),
            classification_loss(outputs, formatted_targets),
            localization_loss(outputs, formatted_targets),
        ]
        print(f"Losses :{losses}")
        running_loss += loss.item() / batch_size
    print(f"Epoch loss {running_loss}")


# %%
