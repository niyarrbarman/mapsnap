import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input
from keras.models import Model
import segmentation_models as sm
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pass