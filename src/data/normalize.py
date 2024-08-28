import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load training data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# Select only numeric columns for scaling
numeric_columns = X_train.select_dtypes(include='number').columns
X_train_numeric = X_train[numeric_columns]
X_test_numeric = X_test[numeric_columns]

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Ensure the output directory exists
os.makedirs('data/scaled', exist_ok=True)

# Save the normalized data
pd.DataFrame(X_train_scaled, columns=numeric_columns).to_csv('data/scaled/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=numeric_columns).to_csv('data/scaled/X_test_scaled.csv', index=False)

# dvc stage add -n normalize \
#               -d src/data/normalize.py -d data/processed \
#               -o data/scaled/scaled \
#               python src/data/normalize.py

# should be explicitly written
# dvc stage add -n normalize \
#               -d src/data/normalize.py \
#               -d data/processed/X_train.csv \
#               -d data/processed/X_test.csv \
#               -o data/scaled/X_train_scaled.csv \
#               -o data/scaled/X_test_scaled.csv \
#               python src/data/normalize.py