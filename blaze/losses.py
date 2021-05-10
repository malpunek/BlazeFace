"""
Based on:
https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation/

both outputs and targets are (N, H, W, anchors_per_cell, [objectness, cx, cy, w, h, class1, class2, ...])
"""
import torch.nn.functional as F

# TODO check these work as they should


def objectness_loss(outputs, targets, noObjectCoeff):
    # loss = outputs[..., 0] - targets[..., 0]
    # obj = (loss[targets[..., 0] == 1] ** 2).sum()
    # noObj = (loss[targets[..., 0] == 0] ** 2).sum()

    o1, t1 = outputs[targets[..., 0] == 1], targets[targets[..., 0] == 1]
    obj = F.mse_loss(o1[..., 0], t1[..., 0])

    o0, t0 = outputs[targets[..., 0] == 0], targets[targets[..., 0] == 0]
    noObj = F.mse_loss(o0[..., 0], t0[..., 0])

    return obj + noObjectCoeff * noObj


def classification_loss(outputs, targets):
    o, t = outputs[targets[..., 0] == 1][..., 5:], targets[targets[..., 0] == 1][
        ..., 5:
    ].argmax(dim=-1)
    return F.cross_entropy(o, t)


def center_loss(outputs, targets):
    o, t = outputs[targets[..., 0] == 1], targets[targets[..., 0] == 1]
    loss_centers = F.mse_loss(o[:, 1:3], t[:, 1:3])
    return loss_centers


def sizes_loss(outputs, targets):
    o, t = outputs[targets[..., 0] == 1], targets[targets[..., 0] == 1]
    loss_sizes = F.mse_loss(o[:, 3:5].abs().sqrt(), t[:, 3:5].abs().sqrt())
    return loss_sizes


def full_loss(
    outputs,
    targets,
    noObjectCoeff=.5,
    centerCoeff=75,
    sizesCoeff=10,
    classificationCoeff=.2,
    allCoeff=5
):
    obj = allCoeff * objectness_loss(outputs, targets, noObjectCoeff)
    cls = allCoeff * classificationCoeff * classification_loss(outputs, targets)
    center = allCoeff * centerCoeff * center_loss(outputs, targets)
    sizes = allCoeff * sizesCoeff * sizes_loss(outputs, targets)
    return (obj + cls + center + sizes), (obj, cls, center, sizes)
