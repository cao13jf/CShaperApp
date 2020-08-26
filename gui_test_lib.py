# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

# ===========================================
# Test the library dependency. You can use `pip install -r requirements.txt` to install all libraries.
# ===========================================
import sys
import os
import glob
import shutil
import pickle
import numpy as np
from scipy import ndimage
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
from treelib import Tree, Node
from skimage import morphology
from skimage.measure import marching_cubes_lewiner, mesh_surface_area

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.bn import BNLayer
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer


# =============================================
# Test whether the parameter can be used by the main function
# =============================================
def main(para):
    print(para)  # <-- All parameters should show here. If it works well, I will add more functions wrt our github
                    # reposiroty at https://github.com/cao13jf/CShaper


if __name__ == "__main__":
    config = {}  # <-- This `config` is supposed to be a dict variable, which can store all parameters input by the user
    main(config)
