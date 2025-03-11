import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# use cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def create_data(samples=1000):
    """Generate input data and corresponding sine values."""
    np.random.seed(0)
    inputs = np.random.uniform(-1, 1, samples).reshape(-1, 1)  # Ensure 2D shape
    targets = np.sin(inputs)
    return inputs, targets

def construct_model():
    """Build a simple neural network."""
    net = Sequential([
        Input(shape=(1,)),  # Single input feature
        Dense(10, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)  # Output a single value
    ])
    net.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return net

def train_network(net, inputs, targets, epochs=1000):
    """Train the model with early stopping."""
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = net.fit(inputs, targets, epochs=epochs, callbacks=[early_stop], verbose=0)  # Suppress logs
    return net, history

def plot_results(trained_net):
    """Plot model predictions vs actual sine function."""
    test_inputs = np.linspace(-1, 1, 1000).reshape(-1, 1)  # Ensure correct input shape
    predictions = trained_net.predict(test_inputs)

    expected_outputs = np.arcsin(test_inputs)  # Correct function comparison
    test_error = trained_net.evaluate(test_inputs, expected_outputs, verbose=0)

    print("Test Loss:", test_error)

    # Save model prediction vs true sine function
    plt.figure(figsize=(6, 4))
    plt.plot(test_inputs, predictions, color='red', label='Model Prediction')
    plt.plot(test_inputs, expected_outputs, color='blue', linestyle='--', label='True sin(x)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title(f'Model Approximation of sin(x) - Test Loss = {test_error:.4f}')
    plt.savefig("model_sin_approximation.png")  # Save image
    plt.show()

def execute_pipeline():
    """Execute the full pipeline: data generation, training, evaluation."""
    data_X, data_y = create_data(1000)  # Use correct input-output pairing
    model = construct_model()
    model, _ = train_network(model, data_y, data_X)
    plot_results(model)
    model.save('model.h5')
    print('Model successfully saved as model.h5')

if __name__ == '__main__':
    execute_pipeline()
