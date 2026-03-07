"""
data.py

This module handles:
- Loading in Data to a Data Loader
- Converts data into Tensor Dataset
- Provides compatibility with the Pytorch based model

Author: Ethan Do and Changli
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing_v2 import DataPreprocessor

def load_and_prepare(config):
    """
    Loads the dataset into a Dataloader and perform preprocessing on it

    Args:
        config: Configuration object containing dataset and training settings
            Expected attributes:
                - DATA_PATH (str): Path to the dataset file
                - TEST_SIZE (float): Fraction of data used for testing
                - RANDOM_SEED (int): Random seed for reproducibility
                - BATCH_SIZE (int): Batch size for DataLoader

    Returns:
        tuple:
            train_loader : DataLoader for training data
            test_loader : DataLoader for test data
            input_size (int): Number of input features
    """
    processor = DataPreprocessor(config.DATA_PATH)
    processor.load_data()
    df = processor.feature_engineering()

    y = df["Delivery_person_Ratings"]
    X = df.drop(columns=["Delivery_person_Ratings"])

    # Split into train and validation data set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Converts the data into tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Builds the datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, test_loader, X_train.shape[1]