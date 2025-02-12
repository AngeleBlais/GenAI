### CNN based GAN
Generator:

-uses dense layers to turn random noise into a sequence of feature embeddings
-applies multi-head self-attention to improve details
-activation with ReLU for hidden layers

Discriminator:
-Converts images into smaller patches and processes them as a sequence.
-Uses self-attention and dense layers to analyze the image structure.
-Multi-head attention helps detect patterns across different patches.
-Activation function: LeakyReLU for hidden layers, Sigmoid for the final decision.

Training:
-Trained on the MNIST dataset to create realistic handwritten digits.
-Uses self-attention to better understand spatial relationships in images.

###Transformer based GAN
-generator
