# =============================================
# 4. HyPerNet: Hybrid ResNet50-EfficientNetB0 with LSTM-GRU Fusion
# =============================================
# 4.1 Model Training

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Concatenate,
    Bidirectional, LSTM, GRU, GlobalAveragePooling2D, Reshape
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
