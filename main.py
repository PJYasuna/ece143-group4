"""
main.py

Author: Ethan Do
"""

from src.config import Config
from src.utils import set_seed, get_device
from src.data import load_and_prepare
from src.trainer import Trainer
from src.model import build_model

def main():
    config = Config()

    set_seed(config.RANDOM_SEED)
    device = get_device(config.DEVICE)

    train_loader, test_loader, input_size = load_and_prepare(config)
  
    model = build_model(config, input_size)

    trainer = Trainer(model, config, device)

    trainer.train(train_loader)

    mae = trainer.evaluate(test_loader)
    print(f"\nFinal Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()