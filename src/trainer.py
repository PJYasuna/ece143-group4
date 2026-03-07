"""
trainer.py

This module handles:
- Creates a Trainer Object to pipeline training steps

Author: Ethan Do and Changli
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

    def train(self, train_loader):
        """
        Trains the model

        Args:
            train_loader : a PyTorch DataLoader object providing training batches

        Returns:
            None
        """
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.EPOCHS):
            self.model.train()
            epoch_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            print(f"Epoch [{epoch+1}/{self.config.EPOCHS}] Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def evaluate(self, test_loader):
        """
        evaluates the model

        Args:
            train_loader : a PyTorch DataLoader object providing training batches

        Returns:
            float: the average MAE over the dataset
            lists: the prediction from the model and the real values
        """
        self.model.eval()
        total_mae = 0

        all_preds = []
        all_actual = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(X_batch)

                total_mae += torch.mean(torch.abs(preds - y_batch)).item()

                all_preds.append(preds.cpu())
                all_actual.append(y_batch.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_actual = torch.cat(all_actual).numpy()

        mae = total_mae / len(test_loader)

        return mae, all_preds, all_actual