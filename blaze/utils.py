import torch
from torchvision.ops import box_convert, box_iou


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
    new_targets = torch.zeros_like(outputs)
    for sample_i, target in enumerate(
        targets
    ):  # target == torch.Tensor(objs, [cx, cy, w, h, class1, class2, ...]
        anchors_boxes = box_convert(anchors, "cxcywh", "xyxy")
        object_boxes = box_convert(target[:, :4], "cxcywh", "xyxy")
        ious = box_iou(object_boxes, anchors_boxes)
        best_boxes = unravel_indices(
            torch.argmax(box_iou, dim=-1), anchors.shape[:3]
        )  #  torch.Tensor(objs, [cell_y, cell_x, anchor#])
        final_objects_i = torch.cat((torch.ones(target.size(0)), target), dim=-1)
        new_targets[
            sample_i, best_boxes[:, 0], best_boxes[:, 1], best_boxes[:, 2], :
        ] = final_objects_i

    return new_targets
