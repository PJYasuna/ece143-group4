"""
config.py

This module handles:
- the configuration of our program allowing flexibility within our model

Author: Ethan Do
"""

class Config:
    DATA_PATH = "data/train.csv"

    TEST_SIZE = 0.1
    RANDOM_SEED = 123

    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 32

    EARLY_STOPPING_PATIENCE = 10

    DEVICE = "cuda" # if gpu allows, "cuda" runs model on cpu leading to faster training
    MODEL_NAME = "mlp" # Options for models are "linear", "mlp". More models can be made later for scalibility.
    HIDDEN_SIZE = 64
