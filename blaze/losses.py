"""
Based on:
https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation/

both outputs and targets are (N, H, W, anchors_per_cell, [objectness, cx, cy, w, h, class1, class2, ...])
"""

# TODO check these work as they should


def objectness_loss(outputs, targets, noObjectCoeff):
    loss = outputs[..., 0] - targets[..., 0]
    obj = (loss[targets[..., 0] == 1] ** 2).sum()
    noObj = (loss[targets[..., 0] == 0] ** 2).sum()
    return obj + noObjectCoeff * noObj


def classification_loss(outputs, targets):
    o, t = outputs[targets[..., 0] == 1], targets[targets[..., 0] == 1]
    loss = o[..., 5:] - t[..., 5:]
    return (loss ** 2).sum()


def localization_loss(outputs, targets):
    o, t = outputs[targets[..., 0] == 1], targets[targets[..., 0] == 1]
    loss_centers = ((o[1:3] - t[1:3]) ** 2).sum()
    loss_sizes = ((o[3:5].abs().sqrt() - t[3:5].abs().sqrt()) ** 2).sum()
    return loss_centers + loss_sizes


def full_loss(outputs, targets, noObjectCoeff=0.5, coordCoeff=5):
    return (
        objectness_loss(outputs, targets, noObjectCoeff)
        + classification_loss(outputs, targets)
        + coordCoeff * localization_loss(outputs, targets)
    )
