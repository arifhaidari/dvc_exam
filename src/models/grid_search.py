import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

# Load data
X_train = pd.read_csv('data/scaled/X_train_scaled.csv') 
y_train = pd.read_csv('data/processed/y_train.csv')

# Model and parameters
model = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train.values.ravel())

# Ensure the output directory exists
os.makedirs('models', exist_ok=True)

# Save best parameters
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')


# dvc stage add -n grid_search \
#               -d src/models/grid_search.py \
#               -d data/scaled/X_train_scaled.csv \
#               -d data/processed/y_train.csv \
#               -o models/best_params.pkl \
#               python src/models/grid_search.py