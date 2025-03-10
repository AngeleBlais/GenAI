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

4. Key Discussion Points
   
What happens for values outside the range [-1,1]?
Les fonctions sinusoidales étant comprises entre -1 et 1 si l'on sort de ces valeurs celles-ci n'auront jamais été vues lors de l'entraînement ce qui entraînera une perte de performance.

What are the implications of approximating inverses in more complex functions?
La prédiction sur des fonctions complexes est d'autant plus difficile puisque une fonction plus complexe peut présenter des discontinuités ce qui ne permet pas une bonne généralisation. Par ailleurs pour la fonction x^2 par exemple il y a plusieurs valeurs de x, le réseau de neurones doit alors s'adapter.

## Part 2: Diffusion Model

Un modèle de diffusion apprend à générer une ou des images en débruitant progressivement une entrée bruitée. Nous allons implémenter un modèle de diffusion simple pour comprendre le processus

Choix du Dataset: MNIST 

3. Le bruit gaussien est ajouté via la fonction apply_noise ci-dessous

![image](https://github.com/user-attachments/assets/d44c26c1-2e60-4d16-a835-bbdc0010c361)

Ce bruit est ajouté à l'image d'entrée pour la perturber, simulant ainsi le processus de dégradation de l'image au fur et à mesure des étapes de diffusion. Il suit une distribution normale ce qui rend les calculs plus simples.

Deux modèle U-Net minimalistes SinusoidalUNet et SimpleUNet sont alors entraînés. Tandis que SimpleUNet utilise un simple embedding pour le temps, SinusoidalUNet utilise un embedding temporel, ce qui permet de mieux capturer la cyclicité du temps. 
Une fois que l'on a ajouté du bruit à l'image et que le bruit a été prédit par le modèle, on peut calculer la mse. 

Résultats de débruitage
Sortie du modèle minimal à chaque étape de temps (texte alternatif)

Sortie du modèle avec l'embarquement temporel sinusoïdal à chaque étape de temps (texte alternatif)

On peut noter que la perte est plus faible dans le modèle avec l'embarquement temporel sinusoïdal, les bords sont mieux définis et nous perdons moins d'informations.

4. Inférence
   
![image](https://github.com/user-attachments/assets/cfcd8079-a629-4891-a676-a0ab39e52564)![image](https://github.com/user-attachments/assets/7da0e98a-a062-410e-a3df-19e024380b3f)

La première image est la sortie du modèle minimal et la deuxième la sortie du modèle sinusoidal, bien que les deux images soient très floues on voit que l'inférence du modèle minimal est meilleure que celle du sinusoidal.

5. Training
   
![image](https://github.com/user-attachments/assets/4dfbf432-1665-4e57-8eee-0eb858cdd3d2)![image](https://github.com/user-attachments/assets/59db3187-ffce-4717-b4b0-5cec0767b519)
Là encore, la première image est la sortie du modèle minimal et la deuxième la sortie du modèle sinusoidal. On remarque également une netteté plus importante à la sortie du modèle sinusoidale.

6. Discussion
   
What happens if we change the noise schedule?
L'ajout du bruit ou sa suppression seront pertubés si l'on change le noise schedule. Un noise schedule mal réglé peut entraîner des images floues ou des résultats de mauvaise qualité. Par exemple, un bruit plus fort aux premiers pas rendra la tâche du modèle plus difficile, car l'information originale sera rapidement détruite. 

How do diffusion models compare to GANs? 
Ces deux modèles génèrent des images. Cependant, les GANs apprennent avec à la fois un générateur et un discriminateur, ce qui permet de produire des images réalistes plus rapidement que les diffusion models. Les modèles de diffusion ajoutent du bruit progressif et apprennent à l’inverser comme nous l'avons vu dans ce tp, ce qui prend plus de temps mais donne souvent des résultats plus nets. Les modèles de diffusion sont plus fiables pour capturer la diversité des données, tandis que les GANs sont plus rapides.

##Final Reflection

How does function inversion relate to diffusion models? 
Nous avons vu qu'un modèle de diffusion apprend à transformer une image complexe en une image bruitée, avec le bruit gaussien. Les diffusion models apprenent également à inverser ce processus via la fonction d'inversion. Cette inversion est similaire à la recherche de l’inverse d’une fonction mathématique où, donné un y, on trouve x tel que 𝑓(𝑥)=𝑦 Les modèles de diffusion apprennent à estimer cette fonction inverse en corrigeant progressivement le bruit.

How does iterative noise removal help generate realistic images? 

Potential applications of diffusion models (e.g., text-to-image generation like Stable 
Diffusion).


