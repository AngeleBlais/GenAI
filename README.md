# Projet TP3 - Transformers

Un Transformer est un modèle d’apprentissage profond puisqu’il repose sur un réseau de neurones et permet de traiter aussi bien du texte, que des images ou même du son. Il prend une séquence de texte en entrée et en génère une autre en sortie. 

Par exemple, pour un modèle de traduction comme mBART, il prend « Les roses rouges sont mes préférées » en français en entrée et génère « The red roses are my favorites » en anglais en sortie. 

Comme nous allons le voir, les transformers reposent sur un mécanisme de self-attention, avec une architecture basée sur l’empilement de couches et une utilisation de l’embedding pour comprendre l’ordre des mots.

## Self-Attention  

Dans une phrase, chaque mot est associé à un token et chaque token a un poids. Dans ce token, il y a trois vecteurs :  
- **Query (Q)** : Détermine les éléments sur lesquels porter l’attention.  
- **Key (K)** : Permet de calculer les scores d’attention.  
- **Value (V)** : Contient les informations à transmettre.  

La formule d’auto-attention est la suivante :

> *L'attention est une technique qui permet à un réseau neuronal de se concentrer sur les parties les plus importantes des données d'entrée. Le self-attention va relier différentes positions d'une même séquence d'entrée pour calculer une représentation de la séquence et donc créer des connexions similaires. La self-attention permet une meilleure parallélisation.*

Prenons la phrase suivante : **"Les roses rouges sont mes préférées"**.  
- **"roses"** porte une attention forte à **"rouges"**, car c’est son adjectif qualificatif.  
- **"préférées"** s’associe à **"roses rouges"** pour comprendre ce qui est préféré.  
- **"sont"** établit la relation entre le sujet **"roses rouges"** et l’attribut **"préférées"**.  

La self-attention permet donc à chaque mot d’interagir avec tous les autres pour pondérer leur importance. Après cette étape, chaque mot possède une **nouvelle représentation contextuelle**, enrichie par les autres mots de la phrase.

## Positional Encoding  

Comme expliqué précédemment, dans un Transformer, les données sont traitées en **parallèle**. Pour comprendre l’ordre des mots dans une phrase, nous utilisons le **positional encoding**. Ce mécanisme s’applique aussi bien aux images qu’aux textes.

### Encodage positionnel absolu  

Ce type d’encodage assigne à **chaque position** d’un mot dans une phrase un **vecteur unique**, indépendant des autres mots. Ce vecteur est généré en utilisant des **fonctions sinus et cosinus**, puis ajouté aux embeddings des mots avant de les passer dans le modèle.

Contrairement aux encodages positionnels appris, l’encodage absolu **ne dépend pas du contexte**.

## Feedforward  

Une fois l’attention terminée, un réseau de neurones **feedforward** est appliqué à chaque token.  

La **self-attention** mélange les informations pour capturer les dépendances entre les mots, mais **les informations ne sont pas transformées**. C’est là que la couche **feedforward** intervient :  
- Elle effectue une **transformation linéaire** suivie d'une **activation non linéaire**.  

Prenons notre exemple : **"Les roses rouges sont mes préférées"**.  
Grâce à **feedforward**, voici les améliorations possibles :  
- **"rouges"** est renforcé pour mieux indiquer qu'il qualifie **"roses"**.  
- **"préférées"** intègre mieux la notion de préférence personnelle grâce à sa connexion avec **"mes"**.  
- **"sont"** est ajusté pour mieux exprimer la relation sujet-attribut.  

## BERT  

**BERT** (*Bidirectional Encoder Representations from Transformers*) est un modèle NLP basé sur les Transformers. Il est conçu pour comprendre le **contexte des mots** en analysant le texte **dans les deux directions**.

Son pré-entraînement repose sur deux tâches :  
1. **Masked Language Modeling (MLM)** : Prédire un mot masqué dans une phrase.  
2. **Next Sentence Prediction (NSP)** : Déterminer si une phrase suit logiquement une autre.  

BERT capture ainsi la **complexité des relations entre les mots** et est utilisé, par exemple, pour **l’analyse des sentiments**.

### Exemple de traitement avec BERT  

Prenons la phrase : **"Les roses rouges sont mes préférées."**  
1. **Tokenisation** avec le tokenizer **WordPiece** :  
   ```plaintext
   ['les', 'roses', 'rouges', 'sont', 'mes', 'pré', 'férées', '.']
