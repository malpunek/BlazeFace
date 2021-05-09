from itertools import product

import geometer as gm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision.ops import box_convert

random = np.random.default_rng(42)


def random_locate(pts: np.array, size=(1280, 720)):
    xt, yt = np.min(pts[:, 0]), np.min(pts[:, 1])
    pts = pts - [xt, yt]
    pts[pts[:, 0] >= size[0]] = size[0] - 1
    pts[pts[:, 1] >= size[1]] = size[1] - 1
    xt, yt = random.integers(0, size[0] - np.max(pts[:, 0])), random.integers(
        0, size[1] - np.max(pts[:, 1])
    )
    return (pts + [xt, yt]).astype(int).flatten().tolist()


def pt2np(*pts):
    return np.stack([pt.normalized_array[:2] for pt in pts])


def get_triangle(
    min_angle=25, size=(1280, 720), equal_sides=True, side_size=(150, 300)
):
    alpha = random.integers(min_angle, (180 - min_angle) / 2)
    beta = alpha
    if not equal_sides:
        alpha = random.integers(min_angle, 180 - 2 * min_angle)
        beta = random.integers(min_angle, 180 - min_angle - alpha)
    side = random.integers(*side_size)
    p1 = gm.Point(0, 0)
    p2 = gm.Point(side, 0)
    s0 = gm.join(p1, p2)
    s1 = gm.rotation(np.deg2rad(alpha)) * s0
    s2 = gm.translation(side, 0) * gm.rotation(np.deg2rad(-beta)) * s0
    p3 = gm.meet(s1, s2)
    final_r = gm.rotation(np.deg2rad(random.integers(0, 360)))
    p2, p3 = final_r * p2, final_r * p3
    pts = pt2np(p1, p2, p3)
    return random_locate(pts, size)


def get_square(side_size=(150, 300), size=(1280, 720)):
    side = random.integers(*side_size)
    r = gm.rotation(np.deg2rad(random.integers(0, 360)))
    p1, p2, p4, p3 = (r * gm.Point(x, y) for x, y in product([0, side], repeat=2))
    return random_locate(pt2np(p1, p2, p3, p4), size)


def get_circle(radious_range=(75, 200), size=(1280, 720)):
    r = random.integers(*radious_range)
    x = random.integers(r, size[0] - r)
    y = random.integers(r, size[1] - r)
    return (x - r, y - r, x + r, y + r)


def random_fill():
    return *random.integers(0, 255, 3), random.integers(140, 255)


def draw_polygon(img):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(5):
        draw.polygon(get_triangle(), fill=random_fill())
        draw.ellipse(get_circle(), fill=random_fill())
        draw.polygon(get_square(), fill=random_fill())
    return np.asarray(img)


def draw_random_fig(img):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, "RGBA")
    fig = random.integers(0, 3)
    if fig == 0:
        draw.polygon(get_triangle(), fill=random_fill())
    elif fig == 1:
        draw.ellipse(get_circle(), fill=random_fill())
    else:
        draw.polygon(get_square(), fill=random_fill())
    return np.asarray(img)


def get_bbox(fig):
    return [min(fig[::2]), min(fig[1::2]), max(fig[::2]), max(fig[1::2])]


def get_figure(img, ftype, kwargs):
    if ftype == 0:
        return get_triangle(size=img.size, **kwargs[ftype])

    elif ftype == 1:
        return get_circle(size=img.size, **kwargs[ftype])

    return get_square(size=img.size, **kwargs[ftype])


def draw_fig(ftype, figure, draw, fill):
    if ftype == 0:
        draw.polygon(figure, fill=fill)

    elif ftype == 1:
        draw.ellipse(figure, fill=fill)

    else:
        draw.polygon(figure, fill=fill)


def random_figures(img, maxFigs):
    n_figs = random.integers(2, maxFigs)
    fig_types = random.integers(0, 3, n_figs)

    kwargs = {
        0: {
            "side_size": (int(min(img.size) / 8), int(min(img.size) / 2)),
        },
        1: {
            "radious_range": (int(min(img.size) / 8), int(min(img.size) / 3)),
        },
        2: {"side_size": (int(min(img.size) / 8), int(min(img.size) / 3))},
    }
    figures = [get_figure(img, ftype, kwargs) for ftype in fig_types]
    return fig_types, figures


def preprocess(img, maxFigs=5):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, "RGBA")

    ftypes, coords = random_figures(img, maxFigs)
    bboxes = [get_bbox(fig) for fig in coords]

    for ftype, figure in zip(ftypes, coords):
        draw_fig(ftype, figure, draw, random_fill())

    targets = torch.zeros(len(ftypes), 4 + 3)
    targets[:, :4] = box_convert(torch.Tensor(bboxes), "xyxy", "cxcywh")
    targets[:, 4:] = F.one_hot(torch.LongTensor(ftypes), num_classes=3)

    return np.asarray(img), targets


def blend(fg, bg, maxFigs=3):
    fg = Image.fromarray(fg)
    draw = ImageDraw.Draw(fg, "RGBA")
    ftypes, coords = random_figures(fg, maxFigs + 1)

    for ftype, figure in zip(ftypes, coords):
        draw_fig(ftype, figure, draw, random_fill())

    mask = Image.new("L", fg.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    for ftype, figure in zip(ftypes, coords):
        draw_fig(ftype, figure, mask_draw, 255)

    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur())

    bg = Image.fromarray(bg)
    bg.paste(fg, (0, 0), mask)

    targets = torch.zeros(len(ftypes), 4 + 3)
    bboxes = [get_bbox(fig) for fig in coords]
    targets[:, :4] = box_convert(torch.Tensor(bboxes), "xyxy", "cxcywh")
    targets[:, 4:] = F.one_hot(torch.LongTensor(ftypes), num_classes=3)

    return bg, targets
