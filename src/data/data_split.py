import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load data
data = pd.read_csv('data/raw/raw.csv')

# Features and target
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# dvc stage add -n split_data \
#               -d src/data/data_split.py \
#               -d data/raw/raw.csv \
#               -o data/processed/X_train.csv \
#               -o data/processed/X_test.csv \
#               -o data/processed/y_train.csv \
#               -o data/processed/y_test.csv \
#               python src/data/data_split.py


