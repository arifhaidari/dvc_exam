# src/data/data_split.py
import pandas as pd
from sklearn.model_selection import train_test_split


# Load data
data = pd.read_csv('data/raw/raw.csv')

# Features and target
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]   # Last column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import os
# Ensure the output directory exists
os.makedirs('data/processed', exist_ok=True)

# Save the splits
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# dvc stage add -n split_data \
#               -d src/data/data_split.py -d data/raw \
#               -o data/processed \
#               python src/data/data_split.py


