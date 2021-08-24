# BlazeFace

An implementation of google's [BlazeFace](https://arxiv.org/pdf/1907.05047.pdf) paper
using Python `3.9.6`.

For an autonomous drone competition my team needed a computer vision model capable of
running fast on SBC's like Raspberry Pi and detecting geometric figures on the ground.
To this end I've created a couple of modules:

* `polygons.py`: A simple module for generating and inpainting geometric figures on images.
* `data.py`: Integration with PyTorch datasets
* `model.py`: The BlazeFace model itself and it's building blocks
* `losses.py`: Implementation of YOLO losses
* `train.py`: The training code
* `quick_eval.py`: Visualizing the results
* `utils.py`


# Instructions

Install `requirements.txt`, change the paths in the code and run using
`python -m blaze.train` or interactively using vscode.