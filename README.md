## Projet TP3
#Transformers
Un Transformer est un modèle d’apprentissage profond puisqu’il repose sur un réseau de neurones et permet de traiter aussi bien du texte, que des images ou même du son. Il prend une séquence de texte en entrée et en génère une autre en sortie. Par exemple pour un modèle de traduction comme mBART, il prend « Les roses rouges sont mes préférées» en français en entrée et génère « The red roses are my favorites » en anglais en sortie. Comme nous allons le voir, les transformers reposent sur un mécanisme de self attention, avec une architecture basée sur l’empilement de couches et une utilisation de l’embedding pour comprendre l’ordre des mots.
#Self Attention  
Dans une phrase, chaque mot est associé à un token et chaque token a un poids. Dans ce token il y a trois vecteurs qui sont la Query qui détermine les éléments sur lesquels porter l’attention, la Key qui permet de calculer les scores d’attention ainsi que la Value v qui contient les informations à transmettre. La formule d’auto-attention est la suivante 
 
L'attention est une technique qui permet à un réseau neuronal de se concentrer sur les parties les plus importantes des données d'entrée. Le self-attention va relier différentes positions d'une même séquence d'entrée pour calculer une représentation de la séquence et donc créer des connexions similaires. La self-attention permet une meilleure parallélisation.
Reprenons la phrase suivante: Les roses rouges sont mes préférées.
-"roses" porte une attention forte à "rouges" car c’est son adjectif qualificatif
-"préférées" va s’associer à "roses rouges" pour comprendre ce qui est préféré
-"sont" établit la relation entre le sujet "roses rouges" et l’attribut "préférées"
La self-attention permet donc à chaque mot d’interagir avec tous les autres pour pondérer leur importance. Après cette étape, chaque mot possède une nouvelle représentation contextuelle, enrichie par les autres mots de la phrase.

#Positional Encoding
Comme expliqué précédemment, dans un transformers, les données sont traitées en parallèle. Pour comprendre l’ordre des mots dans une phrase par exemple, nous allons alors utiliser le positional encoding. Lors du traitement des données le positional encoding permet d’appliquer une notion d’ordre aussi bien aux images qu’aux textes. Il en existe plusieurs types, tel que :
Encodage positionel absolu 
Assigne à chaque position d’un mot dans une phrase un vecteur unique, indépendant des autres mots. Ce vecteur unique est généré en utilisant les fonctions sinus et cosinus. Chaque vecteur est ensuite ajouté aux embeddings des mots avant de les passer dans le transformeur. Contrairement aux encodages positionnels appris, l’encodage absolu ne dépend pas du contexte. 
#Feedforward
Une fois l’attention finie, une reseau de neurones appelé feedforward est appliqué à chaque token. Chaque neurone est connecté à tous les neurones de la couche suivante. Comme vu précédemment, la self-attention permet une meilleure représentation de l’importance entre les mots. Seulement, les informations ne sont pas réellement transformées, c’est là où la couche feedwordward intervient. L’objectif d’une couche feedforward est d'effectuer une transformation linéaire suivi d'une activation non-linéaire.
La self-attention mélange les informations pour capturer les dépendances entre les mots, mais chaque mot a encore besoin d’être transformé individuellement pour affiner son sens.

Plus concrètement si l’on réutilise l’exemple précédent : Les roses rouges sont mes préférées voici les possibilités d’amélioration apportées par feedforward :
-"rouges" pourrait être renforcé pour mieux indiquer qu'il qualifie "roses".
-"préférées" pourrait mieux intégrer la notion de préférence personnelle grâce aux connexions avec "mes".
-"sont" pourrait être ajusté pour mieux exprimer la relation sujet-attribut.
Cela est possible car la couche Feed-Forward applique une transformation non linéaire sur chaque mot individuellement, ce qui affine leur signification.
#Bert
BERT est un modèle de NLP basé sur les Transformers. Il est conçu pour comprendre le contexte des mots en analysant le texte dans les deux directions (avant et après un mot donné). Son pré-entraînement repose sur deux tâches : Masked Language Modeling (MLM). La deuxième est la Next Sentence Prediction (NSP), qui détermine si une phrase suit logiquement une autre. BERT capture ainsi la complexité des relations entre les mots et est utilisé par exemple pour l’analyse des sentiments. Suivons le traitement de la phrase ‘les roses rouges sont mes préférées’ par BERT
La première étape est la tokenisation des mots en sous mots via un tokenizer Wordpiece. La tokenization permet de mieux gérer les variations linguistiques 
Exemple : ['les', 'roses', 'rouges', 'sont', 'mes', 'pré', 'férées', '.'] 
"préférées" est découpé en "pré" et "férées" car il s'agit d'un mot complexe.
BERT ajoute également des tokens spéciaux :
•	[CLS] au début. Ce token a pour but de capturer l'ensemble du contexte de la phrase. Après passage dans BERT, le vecteur associé à CLS contient une représentation globale de la séquence.
