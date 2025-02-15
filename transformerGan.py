import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from GANInterface import GANInterface

class TransformerGan(GANInterface):
    
    @staticmethod
    def build_generator(latent_dim=100):
        inputs = Input(shape=(latent_dim,))
        x = Dense(7 * 7 * 128, activation="relu")(inputs)
        x = Reshape((49, 128))(x)
        position_encoding = tf.range(start=0, limit=49, delta=1)
        position_embedding = layers.Embedding(input_dim=49, output_dim=128)(position_encoding)
        x += position_embedding
        x = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        x = LayerNormalization()(x)
        x = Dense(128, activation="relu")(x)
        x = LayerNormalization()(x)
        x = Reshape((7, 7, 128))(x)
        x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
        x = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)
        return Model(inputs, x, name="Transformer_Generator")

    @staticmethod
    def build_discriminator():
        inputs = Input(shape=(28, 28, 1))
        x = Flatten()(inputs)
        x = Dense(49 * 128, activation="relu")(x)
        x = Reshape((49, 128))(x)
        position_encoding = tf.range(start=0, limit=49, delta=1)
        position_embedding = layers.Embedding(input_dim=49, output_dim=128)(position_encoding)
        x += position_embedding
        x = MultiHeadAttention(num_heads=4, key_dim=128)(x, x)
        x = LayerNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        return Model(inputs, x, name="Transformer_Discriminator")

    @staticmethod
    def build_gan(discriminator, generator, latent_dim):
        gan_input = layers.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        return Model(gan_input, gan_output)

    @staticmethod
    def train(x_train, generator, discriminator, gan, epochs, batch_size=128, latent_dim=100):
        for epoch in range(epochs):
            for _ in range(x_train.shape[0] // batch_size):
                noise = tf.random.normal([batch_size, latent_dim])
                fake_images = generator.predict(noise)
                real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                discriminator.trainable = True
                d_loss_real = discriminator.train_on_batch(real_images, real_labels)
                d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
                discriminator.trainable = False
                misleading_labels = tf.ones((batch_size, 1))
                g_loss = gan.train_on_batch(noise, misleading_labels)
            print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")
