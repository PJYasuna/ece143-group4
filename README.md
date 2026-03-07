# Food Delivery Order Rating Prediction

ECE143 group 4 project

old dataset link - https://www.kaggle.com/datasets/jayjoshi37/daily-food-delivery-orders-and-delivery-time
new dataset link - https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset/data

## File structure
```text
food-delivery-analysis/
│
├── data/
│   ├── dataset.csv (old dataset)
│   ├── test.csv (new dataset)
│   ├── train.csv (new dataset)
│
├── src/
│   ├── preprocessing_v2.py (new dataset preprocessing)
│   ├── visualization.py
│   ├── model.py
│   ├── data.py
│   ├── config.py
│   ├── utils.py
│   ├── trainer.py
│
├── notebook/
│   └── visualization.ipynb #visualizations included in our presentation
│   
├── archive/
│   └── preprocessing.py (old dataset preprocessing)
│   └── changli_modeling (draft of the model)
│
└── README.md
```
## How to run
Clone the repository:

git clone https://github.com/PJYasuna/ece143-group4.git
cd ece143-group4

Run the main program:

python main.py

The script will load the dataset, train the selected model, and output training results.

# Configuration
The project uses a configuration file (config.py) to control model behavior and training parameters.

You can modify values in this file to experiment with different settings.

Example parameters that can be adjusted include:

Dataset path

Training/test split

Learning rate

Batch size

Number of training epochs

Model type

Hidden layer size

Early stopping patience

Device (CPU or GPU)

Example snippet:

DATA_PATH = "data/train.csv"
LEARNING_RATE = 0.002
EPOCHS = 100
BATCH_SIZE = 32
MODEL_NAME = "mlp"

Changing these values allows you to easily test different training configurations without modifying the main code.

## Third party modules
- Python 3.9+
- torch
- pandas
- scikit-learn
- seaborn
- matplotlib
- numpy
Install them using:

pip install torch pandas scikit-learn seaborn matplotlib numpy