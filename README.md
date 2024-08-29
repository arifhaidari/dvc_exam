# DVC & DagsHub Exam - Machine Learning Pipeline

**Full Name**: Arif HAIDARI <br>
**E-mail**: arifhaidari336@gmail.com <br>
**DagsHub URL**: [https://dagshub.com/arifhaidari336/dvc_exam](https://dagshub.com/arifhaidari336/dvc_exam)

## Project Structure

The project is organized as follows:

```bash
├── dvc_exam
│   ├── data
│   │   ├── raw
│   │   ├── processed
│   │   └── scaled
│   ├── metrics
│   ├── models
│   └── src
│       └── data
│       │   ├── data_split.py
│       │   └── normalize.py
│       └── models
│           ├── grid_search.py
│           ├── train.py
│           └── evaluate.py
│
│
├── README.md
├── requirements.txt
└── dvc.yaml
└── dvc.lock
```

## Description

This project demonstrates a machine learning pipeline using DVC (Data Version Control) to manage data, models, and metrics. The pipeline includes steps for data splitting, normalization, hyperparameter tuning, model training, and evaluation.

## Prerequisites

- Python 3.8+
- Git
- DVC
- Virtual environment (optional but recommended)

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://dagshub.com/arifhaidari336/dvc_exam.git
   cd dvc_exam
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv env_dvc
   source env_dvc/bin/activate  # On Windows use `env_dvc\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC**:

   ```bash
   dvc init
   ```

5. **Add the remote storage for DVC**:

   Make sure to configure your DVC remote storage (DagsHub in this case). Modify the commands with your credentials:

   ```bash
   dvc remote add -d myremote https://dagshub.com/arifhaidari336/dvc_exam.dvc
   dvc remote modify myremote --local auth basic
   dvc remote modify myremote --local user <your_username>
   dvc remote modify myremote --local password <your_password>
   ```

## Pipeline Steps

### 1. Data Preparation

**Split the data into train and test sets**:

- `data_split.py` script loads the raw data, splits it into training and testing sets, and saves the processed data to the `data/processed/` directory.

**Command to create the `split_data` stage**:

```bash
dvc stage add -n split_data \
              -d src/data/data_split.py \
              -d data/raw/raw.csv \
              -o data/processed/X_train.csv \
              -o data/processed/X_test.csv \
              -o data/processed/y_train.csv \
              -o data/processed/y_test.csv \
              python src/data/data_split.py
```

### 2. Data Normalization

- `normalize.py` script scales the training and testing data and saves the normalized data to the `data/scaled/` directory.

**Command to create the `normalize` stage**:

```bash
dvc stage add -n normalize \
              -d src/data/normalize.py \
              -d data/processed/X_train.csv \
              -d data/processed/X_test.csv \
              -o data/scaled/X_train_scaled.csv \
              -o data/scaled/X_test_scaled.csv \
              python src/data/normalize.py
```

### 3. Hyperparameter Tuning

- `grid_search.py` performs a grid search on the `GradientBoostingRegressor` model and saves the best parameters to `models/best_params.pkl`.

**Command to create the `grid_search` stage**:

```bash
dvc stage add -n grid_search \
              -d src/data/grid_search.py \
              -d data/scaled/X_train_scaled.csv \
              -d data/processed/y_train.csv \
              -o models/best_params.pkl \
              python src/data/grid_search.py
```

### 4. Model Training

- `train.py` script trains a `GradientBoostingRegressor` model using the best parameters found in the grid search and saves the trained model to `models/gbr_model.pkl`.

**Command to create the `train` stage**:

```bash
dvc stage add -n train \
              -d src/data/train.py \
              -d data/scaled/X_train_scaled.csv \
              -d data/processed/y_train.csv \
              -d models/best_params.pkl \
              -o models/gbr_model.pkl \
              python src/data/train.py
```

### 5. Model Evaluation

- `evaluate.py` evaluates the trained model on the test set and saves the metrics (`MSE` and `R2`) to `metrics/scores.json` and the predictions to `data/predictions.csv`.

**Command to create the `evaluate` stage**:

```bash
dvc stage add -n evaluate \
              -d src/data/evaluate.py \
              -d data/scaled/X_test_scaled.csv \
              -d data/processed/y_test.csv \
              -d models/gbr_model.pkl \
              -o metrics/scores.json \
              -o data/predictions.csv \
              python src/data/evaluate.py
```

### 6. Reproduce the Entire Pipeline

To reproduce the entire pipeline, run:

```bash
dvc repro
```

### 7. Push Data and Model to Remote Storage

To push the data, model, and other outputs to the DVC remote storage, run:

```bash
dvc push
```

## Additional Notes

- **Make sure to set up your DVC remote correctly** using the credentials and storage settings provided by DagsHub or any other platform you're using.
- **Ensure the `data/raw/raw.csv` file exists** in your repository before running `dvc repro`.

By following the steps in this `README.md`, you can reproduce the machine learning pipeline from scratch. You can easily modify the scripts and commands to suit your project requirements. DVC helps you manage and version control your data, models, and experiments effectively.

Feel free to contact me at [arifhaidari336@gmail.com](mailto:arifhaidari336@gmail.com) for any questions or clarifications.

---
