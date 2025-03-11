# Projet TP4
## Part 1: Inversion Function
Sachant que les NN peuvent approximer des fonctions complexes, nous entraînons ici un modèle de manière à inverser la fonction y= sin(x)

1. Generate Dataset

Sachant que l'inverse de sin(y) est arcsin(x), on sait que la fonction arcison a des valeurs comprises entre -1 et 1. 
De ce fait X suit une distribution uniforme comprise entre -1 et 1 et y= sin(x) 

3. Train a Neural Network
   
Une architecture double RELU a été testée pour évaluer la capacité du modèle à apprendre l'inversion
![image](https://github.com/user-attachments/assets/e32bc0f8-7716-4902-8528-86c5f031bd07)

Voici le résultat du modèle avec deux couches cachées ReLU soit Double ReLU. Deux couches cachées de trois neurones chacune avec activation ReLU avec une mse= 0.0023
avec model = train_model(model, y, X), le modèle a été entraîné sur 1000 epochs avec un early stopping

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
Un modèle U-Net minimaliste SimpleUNet est alors entraîné. SimpleUNet utilise un simple embedding pour le temps, ce qui permet de capturer la cyclicité du temps. 
Une fois que l'on a ajouté du bruit à l'image et que le bruit a été prédit pour le modèle, on peut calculer la mse. 

4. Inférence
   
![image](https://github.com/user-attachments/assets/5f36d91e-93f0-4332-9310-80145e607459)

Cette image est la sortie du modèle minimal, on voit que l'inférence de ce modèle n'est pas très bonne puisque l'image est très floue.

5. Training

![image](https://github.com/user-attachments/assets/6d62f2ec-b2cd-417a-b5db-4e8341dccd93)

Là encore, la sortie du modèle minimal montre une diminution rapide de la loss au début suivie d'une stabilisation à une valeur proche de 0.067. Cela montre que le modèle apprend rapidement à estimer le bruit ajouté aux images. 

6. Discussion
   
What happens if we change the noise schedule?

L'ajout du bruit ou sa suppression seront pertubés si l'on change le noise schedule. Un noise schedule mal réglé peut entraîner des images floues ou des résultats de mauvaise qualité. Par exemple, un bruit plus fort aux premiers pas rendra la tâche du modèle plus difficile, car l'information originale sera rapidement détruite. 

How do diffusion models compare to GANs? 

Ces deux modèles génèrent des images. Cependant, les GANs apprennent avec à la fois un générateur et un discriminateur, ce qui permet de produire des images réalistes plus rapidement que les diffusion models. Les modèles de diffusion ajoutent du bruit progressif et apprennent à l’inverser comme nous l'avons vu dans ce tp, ce qui prend plus de temps mais donne souvent des résultats plus nets. Les modèles de diffusion sont plus fiables pour capturer la diversité des données, tandis que les GANs sont plus rapides.

## Final Reflection

How does function inversion relate to diffusion models? 

Nous avons vu qu'un modèle de diffusion apprend à transformer une image complexe en une image bruitée, avec le bruit gaussien. Les diffusion models apprenent également à inverser ce processus via la fonction d'inversion. Les modèles de diffusion apprennent ensuite à estimer cette fonction inverse en corrigeant progressivement le bruit.

How does iterative noise removal help generate realistic images? 

L'élimination progressive du bruit aide à générer des images réalistes car à chaque étape, le modèle estime le bruit présent et le retire progressivement. Cela ramène l'image vers une version plus nette et réaliste, en conservant les détails

Potential applications of diffusion models (e.g., text-to-image generation like Stable 
Diffusion)

Stable Diffusion est un modèle qui transforme du texte en image. On lui donne une description écrite, et il crée une image qui correspond. Ce modèle apprend en observant des millions d'images et leurs descriptions. Il peut générer des paysages, des personnages ou encore des objets. Il est très utilisé pour la création de contenu de jeu vidéos par exemple.


