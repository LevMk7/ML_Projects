from preparations.preparations import *
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import os
import numpy as np

# Searching for the best XGB model hyperparameters

# Base model
xgb_model = XGBRegressor(
    tree_method="hist",  # Specifies the tree construction method, useful for large datasets
    n_jobs=-1,           # Utilize all cores
    random_state=42
)

# Parameter grid for search
param_dist = {
    "learning_rate": [0.05], # Controls how much the model adjusts its predictions after each iteration
    "max_depth": [10],       # Determines the maximum depth of the trees
    "n_estimators": [450],   # Number of trees in the ensemble model
    "subsample": [0.93, 0.95, 0.97], # Controls the fraction of random samples
    "colsample_bytree": [0.6, 0.7, 0.8], # Determines the fraction of features (columns)
    "min_child_weight": [6, 7, 8]        # Determines the minimum weight (sum of instances)
}

# RandomizedSearchCV with progress tracking
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,     # Dictionary containing ranges of hyperparameter values to be searched
    scoring="neg_mean_squared_error",   # Defines the metric
    n_iter=10,                          # Number of random hyperparameter combinations
    cv=2,                               # Number of folds for cross-validation
    verbose=3,                          # Level of detail to be displayed during training
    random_state=42
)

# Start the search
random_search.fit(X_train, y_train)

# Best parameters
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluate with the best parameters
best_model = random_search.best_estimator_
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
output_path = os.path.join(results_dir, "predictions_XGB.csv")
test[['id', 'num_sold']].to_csv(output_path, index=False)

print(f"File saved at: {output_path}")