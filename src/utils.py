"""
utils.py

This module handles:
- Hardware settings for the model

Author: Ethan Do and Changli
"""

import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Sets the seed to ensure reproducility during training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_preference: str):
    """
    Return an available device based on configuration.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")