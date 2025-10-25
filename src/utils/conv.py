import jax
import math
from jax import numpy as np

def conv_shape(H, W, kernel, pad, stride):
    H_out = math.ceil((H - kernel + pad + stride) / stride)
    W_out = math.ceil((W - kernel + pad + stride) / stride)
    return H_out, W_out