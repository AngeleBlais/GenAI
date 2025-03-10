# TP3 - Transformers

Un Transformer est un modèle d’apprentissage profond puisqu’il repose sur un réseau de neurones et permet de traiter aussi bien du texte, que des images ou même du son. Il prend une séquence de texte en entrée et en génère une autre en sortie. 

Par exemple, pour un modèle de traduction comme mBART, il prend « Les roses rouges sont mes préférées » en français en entrée et génère « The red roses are my favorites » en anglais en sortie. 

Comme nous allons le voir, les transformers reposent sur un mécanisme de self-attention, avec une architecture basée sur l’empilement de couches et une utilisation de l’embedding pour comprendre l’ordre des mots.

## Self-Attention  

Dans une phrase, chaque mot est associé à un token et chaque token a un poids. Dans ce token il y a trois vecteurs qui sont la Query qui détermine les éléments sur lesquels porter l’attention, la Key qui permet de calculer les scores d’attention ainsi que la Value v qui contient les informations à transmettre. La formule de la self attention est la suivante 
![image](https://github.com/user-attachments/assets/041c3e59-da1c-496f-9292-eda47d7157b6)

Cette formule permet de voir comment un token interagit avec d’autres tokens. Reprenons la phrase suivante: Les roses rouges sont mes préférées, avec roses le token en cours de traitement :
- "roses" compare son vecteur Q avec les clés K de tous les autres mots 
- obtient des scores de similarité avec rouges, sont par exemple 
- après softmax, "roses" accorde plus d'importance à rouges son adjectif et plus à sont
- valeurs V de rouges sont transférées à roses
  
L'attention est une technique qui permet à un réseau neuronal de se concentrer sur les parties les plus importantes des données d'entrée. Le self-attention va relier différentes positions d'une même séquence d'entrée pour calculer une représentation de la séquence et donc créer des connexions similaires. Après cette étape chaque mot possède une nouvelle représentation contextuelle, grâce à sa relation avec les autres mots de la phrase. Chaque token est traité en même temps, ce qui permet une meilleure parallélisation.

### Lien entre Convolution et Attention

La convolution et l'attention servent à extraire des informations contextuelles avec des mécanismes différents. La convolution utilise un noyau appliqué sur des patches d'image pour produire une combinaison linéaire des valeurs des patches voisins. Elle repose sur des poids fixes, assurant une orthogonalité. Tandis que dans les transformers, l'attention calcule une pondération dynamique via les vecteurs de requête Q, clé K et valeur V, souvent à travers un produit scalaire. Le tout calculé avec une fonction softmax pour la normalisation. Les transformers, en particulier le ViT Vision Transformer, remplacent donc les convolutions par des mécanismes d'attention pour mieux comprendre les relations entre les tokens. Par contre dans les CNN, les convolutions agissent comme un mécanisme d'attention local, ce qui permet de traiter plus efficacement les informations spatiales, le tout avec une structure plus simple et orthonormée. Ainsi,  la convolution utilise une combinaison fixe de voisins locaux, tandis que l'attention dynamique ajuste les pondérations en fonction du contexte global.

## Positional Encoding  

Comme expliqué précédemment, dans un transformers, les données sont traitées en parallèle. Pour comprendre l’ordre des mots dans une phrase par exemple, nous allons alors utiliser le positional encoding. Lors du traitement des données le positional encoding permet d’appliquer une notion d’ordre aussi bien aux images qu’aux textes. 

Chaque token est d'abord converti en un vecteur dense via une couche d’embeddings, seulement ces vecteurs ne contiennent pas leur position. On ajoute donc à chaque vecteur un positional encoding, qui encode la position du token grâce aux fonctions sinusoïdales et cosinus. Cela permet une périodicité contrôlée, le modèle peut ainsi distinguer les positions relatives des tokens. Contrairement à une simple translation dans l’espace vectoriel, le positional encoding est ajouté aux embeddings, ce qui préserve l’information sémantique tout en intégrant la notion d’ordre.

![image](https://github.com/user-attachments/assets/f8f685f6-0964-441f-85c8-048b1e2109ba)


Une mauvaise translation comme dans Word2Vec ne s’applique pas ici, le Positional Encoding n’altère en effet pas directement le sens des mots. Il modifie plutôt la manière dont le modèle apprend les relations de position. La notion de produit scalaire nul entre anciens et nouveaux vecteurs n’est pas un critère pertinent pour analyser la qualité d'un Positional Encoding. En effet, il ne cherche pas à projeter les tokens dans un espace orthogonal, mais plutôt à créer des relations positionnelles.
Pour éviter un effondrement du modèle, il est essentiel de veiller à ce que le Positional Encoding ne crée pas de redondances ou d’ambiguïtés dans la représentation des tokens. Son rôle est de garantir que les positions restent distinguables, sans perturber l’information portée par les embeddings d’origine.

Il en existe plusieurs types, tel que :

### Encodage positionnel appris

Les positions sont apprises par le modèle au lieu d’être définies par une formule fixe, comme dans BERT. Cela nécessite plus de données pour un meilleur apprentissage.

### Encodage positionnel absolu  

Assigne à chaque position d’un mot dans une phrase un vecteur unique, indépendant des autres mots. Ce vecteur unique est généré en utilisant les fonctions sinus et cosinus. Chaque vecteur est ensuite ajouté aux embeddings des mots avant de les passer dans le transformeur. Contrairement aux encodages positionnels appris, l’encodage absolu ne dépend pas du contexte. 

### Encodage positionnel relatif :

Plutôt que d’encoder les positions absolues, ce type d’encodage capture les distances entre tokens, ce qui est utile pour mieux gérer les dépendances à long terme, comme dans Transformer-XL et T5.


## Feedforward  

Une fois l’attention terminée, un réseau de neurones **feedforward** est appliqué à chaque token.  

La **self-attention** mélange les informations pour capturer les dépendances entre les mots, mais **les informations ne sont pas transformées**. C’est là que la couche **feedforward** intervient :  
- Elle effectue une **transformation linéaire** suivie d'une **activation non linéaire**.  

Prenons notre exemple : **"Les roses rouges sont mes préférées"**.  
Grâce à **feedforward**, voici les améliorations possibles :  
- **"rouges"** est renforcé pour mieux indiquer qu'il qualifie **"roses"** 
- **"préférées"** intègre mieux la notion de préférence personnelle grâce à sa connexion avec **"mes"**  
- **"sont"** est ajusté pour mieux exprimer la relation sujet-attribut

## BERT  

BERT est un modèle de NLP basé sur les Transformers. Il est conçu pour comprendre le contexte des mots en analysant le texte dans les deux directions (avant et après un mot donné). Son pré-entraînement repose sur deux tâches : Masked Language Modeling (MLM). La deuxième est la Next Sentence Prediction (NSP), qui détermine si une phrase suit logiquement une autre. BERT capture ainsi la complexité des relations entre les mots et est utilisé par exemple pour l’analyse des sentiments. Suivons le traitement de la phrase ‘les roses rouges sont mes préférées’ par BERT:

La première étape est la tokenisation des mots en sous mots via un tokenizer Wordpiece. La tokenization permet de mieux gérer les variations linguistiques 
1. Exemple  ```
   ['les', 'roses', 'rouges', 'sont', 'mes', 'pré', 'férées', '.']```
préférée est découpé en "pré" et "férées" car il s'agit d'un mot complexe.

BERT ajoute également des tokens spéciaux :
•	[CLS] au début. Ce token a pour but de capturer l'ensemble du contexte de la phrase. Après passage dans BERT, le vecteur associé à CLS contient une représentation globale de la séquence.
•	[SEP] à la fin. SEP permet de séparer en deux phrases si nécessaire

Notre phrase devient donc : ```
  ['[CLS]', 'les', 'roses', 'rouges', 'sont', 'mes', 'pré', '##férées', '.', '[SEP]']```

BERT lit la phrase dans les deux directions en même temps
- rouges comprend qu’il qualifie roses
- préférées comprend que roses rouges est ce qui est préféré.
- mes aide à saisir que la préférence est personnelle
  
Il se distingue par l'attention bidirectionnelle contrairement à d'autres modèles plus traditionnels comme le RNN.

## ViT

Contrairement à BERT, qui traite des tokens de texte, ViT traite des patchs d'image comme des tokens et les analyse avec des mécanismes de self-attention. Le ViT est appliqué à la computer vision plutôt qu'au texte
On a désormais une image contenant la phrase écrite "Les roses rouges sont mes préférées" 
Celle-ci est divisée en plusieurs petits morceaux appelés patches (ex: 16×16 pixels). Une fois l’image divisée en patches, chaque patch est transformé en un vecteur et est traité comme un token, comme si c’était un mot.

Self-attention : 
Dans un contexte orthonormé, ViT se différencie des CNN par l’utilisation du produit scalaire dans le mécanisme d’attention. Si les données sont déjà orthonormées, l’utilisation de self-attention sera tout de même préférée.

## Part 4
 ### 1.	What are the differences in how Transformers process text versus images?
 
Les Transformers tokenisent le texte en sous-mots avec WordPiece. Une fois les mots d’une phrase tokenisés, on y ajoute les embeddings. 
Comme vu précédemment dans le ViT, les transformers divisent les images en patchs. Ces patchs sont ensuite transformés en vecteurs avec des embeddings pour prendre en compte l'ordre.
### 2.	How does the self-attention mechanism adapt to different data modalities?

Comme vu précédemment, le mechansime d’auto attention capture les relations contextuelles entre les mots d’une séquence en attribuant des poids aux tokens. Il va donc chercher la relation syntaxique ou sémantique entre les mots. 
Pour les images, ce ne sont pas des mots mais des patches d’images qui sont analysés. Si l’on reprend l’exemple du ViT, celui-ci divise une image en petits blocs, qui deveiennet des tokens. Et de la même manière, ces tokens vont capturer les relations spatiales entre les tokens via la self attention. 
### 3.	What are the limitations of Transformers, and how can they be mitigated?

Les limitations des Transformers reposent en partie sur leur complexité algorithmique. Cette complexité entraîne une consommation en mémoire et en puissance de calcul élevée. En effet le méchanisme de self attention est très long, car il traite tous les tokens simultanément. Les transformers ont besoin de grandes quantités de données pour avoir une bonne généralisation.  
Pour éviter cela nous pouvons décider de réduire la complexité computationnelle via l’optimisation. Pour se faire nous pouvons utiliser des modèles à attention tels que sparse.


