# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from transformers import (ViTFeatureExtractor, ViTForImageClassification, Trainer,TrainingArguments, DefaultDataCollator)
# from datasets import load_dataset
# from sklearn.metrics import accuracy_score
# import numpy as np
# from PIL import Image

# torch.manual_seed(42)
# dataset= load_dataset("cifar10")
# processor= ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# def preprocess(batch):
#     """
#     Transforme les images PIL en tenseurs normalisés et ajoute la clé 'pixel_values'.
#     """
#     images = [img.convert("RGB") for img in batch["img"]]
#     features = processor(images, return_tensors="np")
#     batch["pixel_values"] = features["pixel_values"]
#     return batch

# # Appliquer la transformation directement sur le dataset
# dataset = dataset.map(preprocess, batched=True, remove_columns=["img"])

# # Définition du modèle ViT pour la classification CIFAR-10 (10 classes)
# model = ViTForImageClassification.from_pretrained(
#     "google/vit-base-patch16-224-in21k",
#     num_labels=10,
#     ignore_mismatched_sizes=True
# )
# model.gradient_checkpointing_enable()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Arguments d'entraînement
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",         # Évaluation à chaque époque
#     save_strategy="epoch",               # Sauvegarde à chaque époque
#     learning_rate=2e-5,                  # Learning rate
#     per_device_train_batch_size=32,      # Batch Training
#     per_device_eval_batch_size=32,       # Batch Eval
#     num_train_epochs=5,                  # Nombre d'époques
#     weight_decay=0.01,                   # Régularisation L2
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     logging_dir="./logs",
#     logging_steps=10,
#     fp16=True,                          
#     dataloader_num_workers=2,           
#     remove_unused_columns=True,
# )

# # Fonction de calcul des métriques
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = np.argmax(pred.predictions, axis=1)
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc}

# # Configuration de l'entraîneur
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     data_collator=DefaultDataCollator(),
#     tokenizer=processor, 
#     compute_metrics=compute_metrics
# )

# # Fonction de prédiction
# def predict_image(image):
#     """
#     Prédit la classe d'une image en utilisant le modèle fine-tuné.
#     """
#     image = image.convert("RGB").resize((224, 224))  # S'assurer que l'image est RGB et redimensionnée
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     predicted_class_idx = logits.argmax(-1).item()
#     return predicted_class_idx
# # Entraînement
# trainer.train()
# model.save_pretrained("vit-cifar10-finetuned")
# processor.save_pretrained("vit-cifar10-finetuned")

# raw_ds = load_dataset("cifar10")["test"]
# sample_image = raw_ds[0]["img"]
# predicted_class = predict_image(sample_image)
# print(f"Classe prédite pour la première image de test CIFAR-10 : {predicted_class}")

# cifar10_classes = {
#     0: "Avion",
#     1: "Automobile",
#     2: "Oiseau",
#     3: "Chat",
#     4: "Cerf",
#     5: "Chien",
#     6: "Grenouille",
#     7: "Cheval",
#     8: "Bateau",
#     9: "Camion"
# }

# raw_ds= load_dataset("cifar10")["test"]
# num_samples= 10

# print("Résultats des prédictions :")
# for i in range(num_samples):
#     image = raw_ds[i]["img"]
#     predicted_class = predict_image(image)
#     class_name = cifar10_classes.get(predicted_class, "Inconnu")
    
#     print(f"Image {i+1}: Classe prédite -> {class_name} (Index: {predicted_class})")

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import (ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments, DefaultDataCollator)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)

# Load CIFAR-10 dataset
dataset = load_dataset("cifar10")

# Load the ViT Feature Extractor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Data Preprocessing Function
def preprocess(batch):
    images = [img.convert("RGB") for img in batch["img"]]
    features = processor(images, return_tensors="np")
    batch["pixel_values"] = features["pixel_values"]
    return batch

# Apply transformations
dataset = dataset.map(preprocess, batched=True, remove_columns=["img"])

# Load the model with correct label count
model = ViTForImageClassification.from_pretrained(
    "./vit-cifar10-finetuned",
    ignore_mismatched_sizes=True
)
model.gradient_checkpointing_enable()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    dataloader_num_workers=2,
    remove_unused_columns=True,
)

# Function to Compute Accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Trainer Configuration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DefaultDataCollator(),
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Train the Model
#trainer.train()
#model.save_pretrained("vit-cifar10-finetuned")
#processor.save_pretrained("vit-cifar10-finetuned")

# CIFAR-10 class names
cifar10_classes = {
    0: "Avion",
    1: "Automobile",
    2: "Oiseau",
    3: "Chat",
    4: "Cerf",
    5: "Chien",
    6: "Grenouille",
    7: "Cheval",
    8: "Bateau",
    9: "Camion"
}

# Function to Plot Training and Validation Accuracy
def plot_training_accuracy():
    metrics = trainer.state.log_history
    epochs, train_acc, val_acc = [], [], []

    for entry in metrics:
        if "epoch" in entry:
            epochs.append(entry["epoch"])
            if "eval_accuracy" in entry:
                val_acc.append(entry["eval_accuracy"])
            if "train_loss" in entry:
                train_acc.append(1 - entry["train_loss"])  # Approximate train accuracy

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("training_accuracy.png")  # Saves the plot as an image


# Function to Compute and Plot Confusion Matrix
def plot_confusion_matrix():
    predictions, labels, _ = trainer.predict(dataset["test"])
    preds = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=cifar10_classes.values(),
                yticklabels=cifar10_classes.values())

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for CIFAR-10 Classification")
    plt.savefig("confusion_matrix.png")  # Saves the plot as an image



# Call the Visualization Functions
plot_training_accuracy()
plot_confusion_matrix()
