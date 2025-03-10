# Projet TP4
## Part 1: Inversion Function
Sachant que les NN peuvent approximer des fonctions complexes, nous entraînons ici un modèle de manière à inverser la fonction y= sin(x)

1. Generate Dataset
Sachant que l'inverse de sin(y) est arcsin(x), on sait que la fonction arcison a des valeurs comprises entre -1 et 1. 
De ce fait X suit une distribution uniforme comprise entre -1 et 1 et y= sin(x) 

2. Train a Neural Network
   
Trois architectures différentes ont été testées pour évaluer la capacité du modèle à apprendre l'inversion
Dans un premier temps un modèle linéaire simple avec une couche cachée de trois neurones avec activation linéaire avec une mse= 0.014
![image](https://github.com/user-attachments/assets/e75a19f9-3d98-4389-91b6-5238ba732ad6)

Ensuite, un modèle avec une seule couche cachée et activation ReLU (Single ReLU) avec une couche cachée de trois neurones avec activation ReLU avec une mse= 0.011
![image](https://github.com/user-attachments/assets/e2a3e1d0-803c-4f29-9103-a833f51db096)

Enfin, un modèle avec deux couches cachées ReLU (Double ReLU) avec 2 couches cachées de 3 neurones chacune avec activation ReLU avec une mse= 0.08
avec model = train_model(model, y, X), le modèle a été entraîné sur 1000 epochs avec un early stopping
![image](https://github.com/user-attachments/assets/0d47aa00-a2f5-489b-b3d5-1db438a80966)

3. La mse du double ReLU est faible ce qui signifie que le réseau de neurones parvient bien à approximer l'inverse de la fonction sinus. Les prédictions du modèles sont plutôt proches des vraies valeurs de arcsin(y). Cependant, pour améliorer ce score on peut encore augmenter le nombre de neurones par couche ou encore enrichir le jeu de données d'entraînement
