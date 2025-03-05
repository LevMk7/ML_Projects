from preparations.preparations import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
if hasattr(tf, 'reset_default_graph'):
    tf.reset_default_graph = tf.compat.v1.reset_default_graph
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
import numpy as np

# Function to build the model for hyperparameter tuning
def build_model(hp):
    model = keras.Sequential()

    # Input layer
    model.add(layers.InputLayer(shape=X_train_scaled.shape[1:]))

    # Hyperparameter tuning for hidden layers
    for i in range(hp.Int('num_layers', 1, 4)):  # Number of hidden layers (1 to 4)
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),  # Number of units
            activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh', 'sigmoid']),  # Activation function
        ))

    # Output layer
    model.add(layers.Dense(1))

    # Compile the model with hyperparameter tuning for learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return model

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the RandomSearch tuner
tuner = RandomSearch(
    build_model,
    objective='val_mean_squared_error',
    max_trials=10,  # Number of trials
    executions_per_trial=1,  # Number of runs per trial
    directory='my_dir',
    project_name='keras_tuning',
    overwrite=True
)

# Start the hyperparameter search
tuner.search(X_train_scaled, y_train, epochs=100, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])

# Get the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hp.values}")

# Build the best model with the best hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on validation data
y_val_pred = best_model.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error (MSE) on validation: {mse}")

# Path to the results folder
results_dir = os.path.join("../results")
os.makedirs(results_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Prepare test data
X_test = test[features]
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_test_pred = best_model.predict(X_test_scaled)

# Handle negative values, round them, and convert to integers
y_test_pred = np.round(y_test_pred).astype(int)  # Convert to integers
y_test_pred = np.maximum(y_test_pred, 0)        # Replace negative values with 0

# Add predictions to the test DataFrame
test['num_sold'] = y_test_pred

# Save only the required columns (id and num_sold) to the CSV file
output_path = os.path.join(results_dir, "predictions_NN.csv")
test[['id', 'num_sold']].to_csv(output_path, index=False)

print(f"File saved at: {output_path}")