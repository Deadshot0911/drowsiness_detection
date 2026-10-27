"""
Tier 2 Research: CNN + Hidden Markov Model (HMM) for Drowsiness Detection
This was developed as a proof-of-concept to detect temporal sequences of eye states.
It was ultimately not deployed to the final apps due to CPU/Android latency constraints.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
import pickle
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
IMG_SIZE = 224
MAX_FRAMES = 60
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# NOTE: Update this path to where your dataset was located in Colab
DATASET_ROOT = "/content/dataset" 

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_sequences(root):
    sequences, labels = [], []
    for cls_name, label in [("drowsy", 1), ("notdrowsy", 0)]:
        cls_dir = os.path.join(root, cls_name)
        for sub in os.listdir(cls_dir):
            seq_dir = os.path.join(cls_dir, sub)
            frames = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))])
            if len(frames) > 5:
                sequences.append([os.path.join(seq_dir, f) for f in frames[:MAX_FRAMES]])
                labels.append(label)
    return sequences, np.array(labels)

# ==========================================
# 3. CNN FEATURE EXTRACTOR
# ==========================================
def build_cnn_feature_extractor():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    features = layers.Dense(128, activation='relu', name='feature_vector')(x)
    return models.Model(inputs, features, name='CNN_Feature_Extractor')

# ==========================================
# 4. HMM TRAINING
# ==========================================
def train_hmm(feature_sequences, n_components=4):
    X = np.concatenate(feature_sequences, axis=0)
    lengths = [len(seq) for seq in feature_sequences]
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    model.fit(X, lengths)
    return model

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Tier 2 Architecture (CNN + HMM) ready for research.")
    cnn = build_cnn_feature_extractor()
    cnn.summary()
    print("To run the full training, provide a valid DATASET_ROOT path.")
