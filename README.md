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


# Projet TP3
pos encoding: token est converti en vecteur dense via une couche d'embeddings, prendre la pos du token dans la sequence, prendre cette info dans un espace vectoriel, veteur complet information est forcémment conservée/ hypothése de periodicité avec sin/cosin= periodicité assez dynamiqueMAIS pos encoding ajoute donc un déplacement, translater l'information déplacée dans l'espace vectoriel! MAIS quelles sont les consequences d'une telle translation dans l'espace vectoriel? bibliotheque word2vec mauvais déplacement peut changer le sens du mot, trouver une condition ou la translation n'est pas possible, faire attention à l'entrainement et l'organisation du jeu de données , pour cela il faut faire une analyse des corrélations EVITER les possibilités d'effondrement du modele, ORGANISER le jeu de données pour garantir l'othogonalité ou l'indépendance, produit scalaire si il est egal à 0 translation(nous voulons donc qu'il soit different de 0 pour éviter la translation) 
Token → Couche d'Embedding → Vecteur Dense
Ces vecteurs denses sont ensuite utilisés comme entrées pour les mécanismes d'attention et les autres composants du modèle Transformer.
convolution: somme pondérée sur une fenêtre glissante d'un signal ou d'une image (sur input) 

2) dans la convolution on a un noyau de convolution, self attention, SANS la softmax, RGB ajouter un cls, un token de classification, TRANSFORMERS: attention layer pour la softmax: Wq * Wk * Wv query keys et valeurs, la dimension des clefs est optionnelle dans ce cas on prend le v (si on prend pas cas du CROSS ATTENTION et non du SELF: (embeddong(reste une base /projection avec subjectivité peut supperposer deux points))si j'ai vi j'ai ki fonc ki keys pas si important WkUTILISE POUR DEFINIR L'ORTHOGONALITE, si ON UTILISE L'ACP on peut se débarraser de la corrélation et après PRODUIT SCALAIRE !! DONC UTILISATION DE LA MATRICE Wk QUAND ON EST PAS DANS UN REPERE ORTHONORME !!
