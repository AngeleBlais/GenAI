import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification

# Chargement du dataset IMDb
(train_reviews, train_labels), (test_reviews, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Récupération du dictionnaire d'index des mots
word_dict = tf.keras.datasets.imdb.get_word_index()
word_to_index = {idx + 3: word for word, idx in word_dict.items()}
word_to_index.update({0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"})

def review_to_text(sequence):
    """
    Convertit une séquence d'indices en texte lisible
    """
    return " ".join([word_to_index.get(idx, "?") for idx in sequence])

# Conversion des séquences en texte
train_sentences = [review_to_text(seq) for seq in train_reviews]
test_sentences = [review_to_text(seq) for seq in test_reviews]

# Création des DataFrames
train_data = pd.DataFrame({'texte': train_sentences, 'étiquette': train_labels})
test_data = pd.DataFrame({'texte': test_sentences, 'étiquette': test_labels})

# Initialisation du tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, labels, max_len=256):
    """
    Transforme les textes en tokens adaptés au modèle BERT
    """
    tokens = tokenizer.batch_encode_plus(
        texts, add_special_tokens=True, max_length=max_len,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='tf'
    )
    return {
        'input_ids': tokens['input_ids'],
        'attention_mask': tokens['attention_mask'],
        'labels': tf.convert_to_tensor(labels, dtype=tf.int32)
    }

# Prétraitement des données
dataset_train = tokenize_texts(train_data['texte'].tolist(), train_data['étiquette'].tolist())
dataset_test = tokenize_texts(test_data['texte'].tolist(), test_data['étiquette'].tolist())

# Création des datasets TensorFlow
train_ds = tf.data.Dataset.from_tensor_slices(dataset_train).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(dataset_test).batch(32)

# Initialisation du modèle BERT
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optim = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bert_model.compile(optimizer=optim, loss=loss_function, metrics=['accuracy'])

# Entraînement du modèle
training_history = bert_model.fit(train_ds, epochs=3, validation_data=test_ds)

# Prédictions sur le dataset de test
preds = bert_model.predict(test_ds)
predicted_labels = np.argmax(preds.logits, axis=1)

# Affichage des performances
print("Précision :", accuracy_score(dataset_test["labels"].numpy(), predicted_labels))
print(classification_report(dataset_test["labels"].numpy(), predicted_labels))

def sentiment_prediction(phrase):
    """
    Prédit le sentiment d'une critique de film en utilisant BERT
    """
    encodage = tokenizer.encode_plus(
        phrase, add_special_tokens=True, max_length=256,
        padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='tf'
    )
    logits = bert_model(encodage)[0]
    prediction = np.argmax(logits, axis=1)[0]
    return "Positif" if prediction == 1 else "Négatif"

# Test de la fonction
print(sentiment_prediction("I love this GenAI course!"))  
print(sentiment_prediction("This Lab was terribly complicated."))  

# Sauvegarde du modèle et du tokenizer
bert_model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_tokenizer")
