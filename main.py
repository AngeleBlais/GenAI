import argparse
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from cnnGan import cnnGan
from tranformerGan import transformerGan

# Check for available GPUs
def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs detected by TensorFlow:")
        for gpu in gpus:
            print("  ", gpu)
    else:
        print("No GPU detected by TensorFlow. Running on CPU.")
    print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

# GAN Model Dictionary
gan_dict = {
    "cnn": cnnGan,
    "transformer": transformerGan
}

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='GAN Training Script')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='latent dimension')
    parser.add_argument('--dataset', type=str, default=None, help='dataset to use')
    parser.add_argument('--model', type=str, default="cnn", help='model to use')
    parser.add_argument('--output', type=str, default="output.png", help='output file')
    parser.add_argument('--n_images', type=int, default=10, help='number of images to generate')
    return parser.parse_args()
