# UI-Element-Detection
Implementing Faster RCNN via detectron2 to detect UI Elements


## Overview

This is a custom implementation of Detectron2 Fast-RCNN, which can find elements - buttons, titles, different input fields and much more - in any web design prototype or web UI image. 

Furthermore, it shows that this functionality and its value can be served online (if hosted on a webserver) and can be used by interested users (designers) in the browser.

## Installation

### Install Cuda and cuDNN

Below is the setup process. Assuming you have Ubuntu 20.04 installed with an Nvidia GPU available.

To make pytorch work with GPU, it is necessary to install CUDA and cuDNN. If working from Ubuntu 20.04 the [following guide](https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04) is suitable.
  

**After installing both CUDA and cuDNN check CUDA and cuDNN are version 10.1 and 7.6.5 respectively:**

**CUDA:**

    $ nvcc -V

**cuDNN:**

get where cuda installed:

    $ whereis cuda
then navigate to that folder's /include folder and: 

    $ cat cudnn.h | grep CUDNN_MAJOR -A 2

  
  

### Activate conda venv

ensure that conda is initialised with Python 3.8.3 (set it within conda environment creation). 

See [cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) !

With conda environment activated:

    $ pip install tensorflow-gpu==2.2.0

To install correct pytorch and torchvision do this with conda activated environment:

    $ conda install -c pytorch torchvision cudatoolkit=10.1 pytorch

Check that GPU is available and working. In virtual environment `$ python`:

    import torch
    torch.cuda.current_device()
    torch.cuda.get_device_name(0)

Then install some Detectron2 dependencies:

    $ pip install cython pyyaml==5.1
    $ pip install pycocotools
    $ pip install opencv-python

Then install Detectron2:

    $ python -m pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

For more information see also Detectron2 [installation guide](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

## Training
The annotations are provided in data folder, for the images please contact me. Change the dataset paths in **train_detectron2.py** and run.
```
python train_detectron2.py
```

## Inference
The weights are too large to store in the repo so contact me for the same, or train on dataset and save you own weights.
To get output on single image run:
```
python detect_elements.py --image <image_path>
```

Future Work:
1. Create Restfull API to test the model on web with frontend.
2. Add more element classes.
3. Develop script to create full high fidelity design. 

