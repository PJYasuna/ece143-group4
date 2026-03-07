"""
model.py

This module handles:
- creating our model based on config

Author: Ethan Do and Changli
"""
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.network = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

def build_model(config, input_size: int):
    """
    Function to build model based on the configuration

    Args:
        config: Configuration object containing the model's settings
        input_size (int): Number of input features

    Returns:
        torch.nn.Module: Instantiated model
    """

    if config.MODEL_NAME == "linear":
        return LinearRegression(input_size)

    elif config.MODEL_NAME == "mlp":
        return SimpleMLP(input_size, config.HIDDEN_SIZE)

    else:
        raise ValueError(f"Unknown model type: {config.MODEL_NAME}")