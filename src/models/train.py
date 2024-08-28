# src/models/train.py

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# Load data
X_train = pd.read_csv('data/scaled/X_train_scaled.csv')  # Updated path to scaled data
y_train = pd.read_csv('data/processed/y_train.csv')

# Load best parameters
best_params = joblib.load('models/best_params.pkl')

# Train the model
model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train.values.ravel())

# Ensure the output directory exists
os.makedirs('models', exist_ok=True)

# Save the trained model
joblib.dump(model, 'models/gbr_model.pkl')


# dvc stage add -n train \
#               -d src/models/train.py \
#               -d data/scaled/X_train_scaled.csv \
#               -d data/processed/y_train.csv \
#               -d models/best_params.pkl \
#               -o models/gbr_model.pkl \
#               python src/models/train.py