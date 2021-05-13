import torch
import torchvision as tv
from torchvision.ops import box_convert, box_iou

from .data import Frames


# https://github.com/pytorch/pytorch/issues/35674
def unravel_indices(
    indices: torch.LongTensor,
    shape: tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def targets_to_batch(targets, outputs, anchors):
    """
    Args:
        outputs: (N, H, W, anchors_per_cell, [obj, x, y, w, h, class1, class2, ...])
        targets: list(torch.Tensor(objs, [x, y, w, h, class1, class2, ...])), len(list) == N
        anchors: torch.Tensor(H, W, anchors_per_cell, 4)
    """
    new_targets = torch.zeros_like(outputs).cpu()
    for sample_i, target in enumerate(
        targets
    ):  # target == torch.Tensor(objs, [cx, cy, w, h, class1, class2, ...]
        anchors_boxes = box_convert(anchors, "cxcywh", "xyxy").reshape(-1, 4)
        object_boxes = box_convert(target[:, :4], "cxcywh", "xyxy")
        ious = box_iou(object_boxes, anchors_boxes)
        best_boxes = unravel_indices(
            torch.argmax(ious, dim=-1), anchors.shape[:3]
        )  # torch.Tensor(objs, [cell_y, cell_x,ta anchor#])
        final_objects_i = torch.cat((torch.ones(target.size(0), 1), target), dim=-1)
        new_targets[
            sample_i, best_boxes[:, 0], best_boxes[:, 1], best_boxes[:, 2], :
        ] = final_objects_i

    return new_targets


def get_eval_sample():
    data = Frames("/data/droniada/eval256_144.mp4")
    trans = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
            tv.transforms.Resize((72, 128)),
        ]
    )
    eframes = [trans(data.random()) for _ in range(1)]
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
