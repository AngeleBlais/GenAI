import argparse
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


def fetch_imdb_data(vocabulary_size=10000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocabulary_size)
    return x_train, y_train, x_test, y_test

def load_tokenizer(bert_model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(bert_model_name)

def tokenize_and_pad(text_data, tokenizer, max_len=256):
    word_index = tf.keras.datasets.imdb.get_word_index()
    index_to_word = {i + 3: word for word, i in word_index.items()}
    index_to_word.update({0: "[PAD]", 1: "[START]", 2: "[UNK]", 3: "[UNUSED]"})
    
    text_reviews = [" ".join([index_to_word.get(i, "[UNK]") for i in review]) for review in text_data]
    
    return tokenizer(text_reviews, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

def create_data_loader(encoded_data, labels, batch_size, shuffle_data=True):
    data = TensorDataset(encoded_data["input_ids"], encoded_data["attention_mask"], torch.tensor(labels))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle_data)

def initialize_model(bert_model_name="bert-base-uncased", num_labels=2):
    return BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)

def train_bert_model(model, train_loader, validation_loader, epochs, learning_rate, device):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    
    train_losses, validation_accuracies = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct_preds, total_preds = 0, 0
        with torch.no_grad():
            for batch in validation_loader:
                input_ids, attention_mask, labels = [item.to(device) for item in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct_preds += (predictions == labels).sum().item()
                total_preds += labels.size(0)
        
        validation_accuracy = correct_preds / total_preds
        validation_accuracies.append(validation_accuracy)
        
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {validation_accuracy:.4f}")
    
    return train_losses, validation_accuracies

def evaluate_bert_model(model, test_loader, device):
    model.eval()
    all_predictions, all_true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    test_accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    return test_accuracy, all_predictions, all_true_labels

def save_trained_model(model, save_directory):
    model.save_pretrained(save_directory)
    print(f"Model saved at {save_directory}")

def plot_training_results(train_losses, validation_accuracies, true_labels, predicted_labels):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.title("Train Loss & Validation Accuracy")
    plt.show()

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def perform_inference(input_text, model_path="./saved_bert_imdb"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    encoded_input = tokenizer(input_text, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    class_names = ["Negative", "Positive"]
    
    print(f"The text '{input_text}' is classified as {class_names[predicted_class]} with a probability of {probabilities[0][predicted_class].item():.4f}")

def main_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = fetch_imdb_data(vocabulary_size=10000)
    tokenizer = load_tokenizer()
    
    train_encodings = tokenize_and_pad(x_train, tokenizer, max_len=args.max_length)
    test_encodings = tokenize_and_pad(x_test, tokenizer, max_len=args.max_length)
    
    train_loader = create_data_loader(train_encodings, y_train, args.batch_size)
    full_test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(y_test))
    
    val_size = len(full_test_dataset) // 2
    test_size = len(full_test_dataset) - val_size
    val_dataset, test_dataset = random_split(full_test_dataset, [val_size, test_size])
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = initialize_model()
    train_losses, validation_accuracies = train_bert_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    
    save_trained_model(model, args.save_path)
    
    test_accuracy, predicted_labels, true_labels = evaluate_bert_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    plot_training_results(train_losses, validation_accuracies, true_labels, predicted_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT on IMDb sentiment classification")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimization")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    parser.add_argument("--save_path", type=str, default="./saved_bert_imdb", help="Directory to save the trained model")
    parser.add_argument("--inference", action="store_true", help="Run inference on a sample text")
    parser.add_argument("--input_text", type=str, default="This movie was amazing!", help="Text for inference")
    
    args = parser.parse_args()
    
    if args.inference:
        perform_inference(args.input_text)
    else:
        main_pipeline(args)
