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

# Train GAN Model
def train_gan(args):
    if args.model not in gan_dict:
        print("Invalid model")
        return
    
    gan_class = gan_dict[args.model]
    x_train = commonGan.load_data()
    
    generator = gan_class.build_generator()
    print(generator.summary())
    
    discriminator = gan_class.build_discriminator()
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), 
                          loss="binary_crossentropy", 
                          metrics=["accuracy"])
    discriminator.trainable = False
    
    gan = gan_class.build_gan(discriminator, generator, args.latent_dim)
    gan.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), 
                loss="binary_crossentropy")
    
    gan_class.train(x_train, generator, discriminator, gan, 
                    args.epochs, args.batch_size, args.latent_dim)
    
    commonGan.generate_images(generator, args.n_images, args.latent_dim)

# Main function
def main():
    check_gpus()
    args = parse_arguments()
    train_gan(args)

if __name__ == '__main__':
    main()
