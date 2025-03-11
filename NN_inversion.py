import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def create_data(samples=1000):
    np.random.seed(0)
    inputs = np.random.uniform(-5, 5, samples)
    targets = np.sin(inputs)
    return inputs, targets

def construct_model():
    net = Sequential([
        Input(shape=(1,)),
        Dense(3, activation="relu"),
        Dense(1)
    ])
    net.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return net

def train_network(net, inputs, targets, epochs=1000):
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = net.fit(inputs, targets, epochs=epochs, callbacks=[early_stop])
    return net, history

def plot_results(trained_net):
    test_inputs = np.linspace(-1, 1, 1000)
    predictions = trained_net.predict(test_inputs)
    expected_outputs = np.sin(test_inputs)
    test_error = trained_net.evaluate(test_inputs, expected_outputs)
    print("Test Loss:", test_error)
    
    plt.plot(test_inputs, predictions, color='red', label='Model Prediction')
    plt.plot(test_inputs, np.arcsin(test_inputs), color='green', linestyle='-', label='True arcsin')
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('x')
    plt.title(f'Model Approximation of arcsin(y) - Test Loss = {test_error}')
    plt.show()

def execute_pipeline():
    data_X, data_y = create_data(100)
    model = construct_model()
    model, _ = train_network(model, data_y, data_X)
    plot_results(model)
    model.save('model.h5')
    print('Model successfully saved as model.h5')

if __name__ == '__main__':
    execute_pipeline()
