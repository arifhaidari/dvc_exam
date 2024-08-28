import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

# Load data
X_test = pd.read_csv('data/scaled/X_test_scaled.csv') 
y_test = pd.read_csv('data/processed/y_test.csv')

# Load model
model = joblib.load('models/gbr_model.pkl')

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Ensure output directories exist
os.makedirs('metrics', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Save metrics
with open('metrics/scores.json', 'w') as f:
    json.dump({'MSE': mse, 'R2': r2}, f)

# Save predictions
pd.DataFrame(predictions, columns=['Prediction']).to_csv('data/predictions.csv', index=False)


# dvc stage add -n evaluate \
#               -d src/models/evaluate.py \
#               -d data/scaled/X_test_scaled.csv \
#               -d data/processed/y_test.csv \
#               -d models/gbr_model.pkl \
#               -o metrics/scores.json \
#               -o data/predictions.csv \
#               python src/models/evaluate.py