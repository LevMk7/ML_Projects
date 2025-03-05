from preparations.preparations import *
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna
import os
import numpy as np

# Searching for the best LGB model hyperparameters

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'device_type': 'gpu',       # Use GPU
        'num_threads': -1,           # Utilize all cores
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 10, 20),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'seed': 100
    }
    model = LGBMRegressor(**param)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Number of iterations
print(f"Best parameters: {study.best_params}")

# Evaluate with the best parameters
best_model = LGBMRegressor(**study.best_params)
best_model.fit(X_train, y_train)
y_val_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error (MSE) on validation: {mse}")

# Path to the results folder
results_dir = os.path.join("../results")
os.makedirs(results_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Prepare test data
X_test = test[features]

# Make predictions
y_test_pred = best_model.predict(X_test)

# Handle negative values, round them, and convert to integers
y_test_pred = np.round(y_test_pred).astype(int)  # Convert to integers
y_test_pred = np.maximum(y_test_pred, 0)        # Replace negative values with 0

# Add predictions to the test DataFrame
test['num_sold'] = y_test_pred

# Save only the required columns (id and num_sold) to the CSV file
output_path = os.path.join(results_dir, "predictions_LGB.csv")
test[['id', 'num_sold']].to_csv(output_path, index=False)

print(f"File saved at: {output_path}")