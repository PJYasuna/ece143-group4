"""
main.py

Author: Ethan Do amd Changli
"""

from src.config import Config
from src.utils import set_seed, get_device
from src.data import load_and_prepare
from src.trainer import Trainer
from src.model import build_model
import matplotlib.pyplot as plt
import pandas as pd

def main():
    config = Config()

    set_seed(config.RANDOM_SEED)
    device = get_device(config.DEVICE)

    train_loader, test_loader, input_size = load_and_prepare(config)
  
    model = build_model(config, input_size)

    trainer = Trainer(model, config, device)

    trainer.train(train_loader)

    mae, preds, actual = trainer.evaluate(test_loader)
    print(f"\nFinal Test MAE: {mae:.4f}")
    mae, preds, actual = trainer.evaluate(test_loader)
    plt.figure()
    plt.scatter(actual, preds)

    min_val = min(actual.min(), preds.min())
    max_val = max(actual.max(), preds.max())

    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.show()

    results = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": preds.flatten()
    })

    results.to_csv(f"results/data/{config.MODEL_NAME}_predictions.csv", index=False)

if __name__ == "__main__":
    main()