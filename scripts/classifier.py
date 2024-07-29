import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import embedder as embedder

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# handles text data and convert it to the required format for BERT
class QueryDataset(Dataset):
    def __init__(self, queries, labels):
        """
        Initialize the dataset with texts, labels, tokenizer, and maximum length.
        - texts: List of input text strings.
        - labels: List of corresponding labels.
        - max_len: Maximum length of tokenized input sequences.
        """

        self.queries = queries
        self.labels = labels
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.queries)
    
    def __getitem__(self, item):
        """
        Get a single sample of text, tokenize it, and reutrn the required inputs for BERT.
        - item: Index of the sample
        """
        query = self.queries[item]
        label = self.labels[item]
        
        embedding,_ = embedder.embedder(query,128).squeeze(0)

        return {
            'query': query,
            'embedding': embedding,
            'label': torch.tensor(label)
        }
    

class SimpleClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super(SimpleClassifier, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, num_labels)

    def forward(self, x):
        return self.linear(x)
        




def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    """
    Train the model for one epoch.
    - model: The model to train.
    - data_loader: DataLoader for training data.
    - loss_fn: Loss function.
    - optimizer: Optimizer.
    - device: Device to train on
    - n_examples: Number of example in the dataset
    """

    model = model.train() # train mdoe
    losses = []
    correct_pred = 0

    for step, batch in enumerate(data_loader):

        embeddings = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()

        # forward pass
        outputs = model(embeddings)
        # print("train label: ", labels)
        # print("train output:", outputs)
        loss = loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1)

        correct_pred += torch.sum(preds == labels)
        losses.append(loss.item())

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Step {step}/{len(data_loader)}: Loss = {loss.item()}")

    return (correct_pred.double()/n_examples), np.mean(losses)




def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    Evaluate the model on validation set.
    - model: The model to evaluate.
    - data_loader: DataLoader for validation data.
    - loss_fn: Loss function.
    - device: Device to eval on
    - n_examples: Number of examples in the dataset
    """

    model = model.eval() # eval mode
    losses = []
    correct_pred = 0

    with torch.no_grad():
        for batch in data_loader:

            embeddings = batch["embedding"].to(device)
            labels = batch["label"].to(device)
            
            # forward pass
            outputs = model(embeddings)
            # print("val label:", labels)
            # print("val output: ", outputs)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)

            correct_pred += torch.sum(preds == labels)
            losses.append(loss.item())



    return (correct_pred.double()/n_examples), np.mean(losses)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

def predict(model, query, max_length):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        embedding,_ = embedder.embedder(query, 128)
        embedding = torch.from_numpy(embedding.squeeze(0)).to(device)
        logits = model(embedding.unsqueeze(0))  # Add batch dimension
        probabilities = softmax(logits)  # Apply softmax
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities

def main():

    train_queries = [
        # General queries
        "Explain the concept of attention in transformers.",
        "What is the purpose of the residual connections in ResNet?",
        "Summarize the 'Attention is All You Need' paper.",
        "Describe the architecture of a convolutional neural network.",
        "What are the main contributions of the 'Deep Residual Learning for Image Recognition' paper?",
        "How does the analytic convolutional layer work?",
        "What are the benefits of using radiomic features in BRAINMETDETECT?",
        "Explain deep probability aggregation clustering.",
        "What is the FedMRL approach for medical imaging?",
        "How does reinforcement learning help in graph anomaly detection?",
        "What is periodic agent-state based Q-learning?",
        "Describe the adaptive knowledge matching in personalized federated domain-incremental learning.",
        "How is machine learning used for spacecraft thermal simulation?",
        "What is attention?",
        "Explain the basics of convolutional neural networks.",
        "What is reinforcement learning?",
        "What are the key points of the 'Attention is All You Need' paper?",
        "How does the analytic convolutional layer differ from traditional layers?",
        "What are the key findings of the BRAINMETDETECT study?",
        "Explain the concept of deep probability aggregation clustering.",
        "How does FedMRL improve medical imaging?",
        "What are the challenges in graph anomaly detection with noisy labels?",
        "How does periodic agent-state based Q-learning work?",
        "What are the main features of personalized federated domain-incremental learning?",
        "What is the role of physics-informed machine learning in spacecraft thermal simulation?",
        "What is a neural network?",
        "Explain the concept of machine learning.",
        "What is a convolutional layer?",
        "What are the different types of machine learning algorithms?",
        "Describe the structure of a transformer model.",
        "What are the applications of deep learning in medical imaging?",
        "How does federated learning work?",
        "What is the significance of the 'Deep Residual Learning for Image Recognition' paper?",
        "Explain the role of radiomic features in predicting primary tumor from brain metastasis MRI data.",
        "What is the main idea behind periodic agent-state based Q-learning for POMDPs?",
        "How does reinforcement learning apply to graph anomaly detection?",
        "What are the key advantages of using physics-informed machine learning for spacecraft thermal simulation?",
        "Explain the concept of adaptive knowledge matching in federated learning.",
        #38
        # Specific queries
        "How many layers are there in ResNet-50?",
        "What is the email of the primary contact for the 'Deep Residual Learning for Image Recognition' paper?",
        "How many clusters does the deep probability aggregation clustering method create?",
        "What is the MRI data size used in BRAINMETDETECT?",
        "What are the parameters used in the periodic agent-state based Q-learning?",
        "What is the sample size used in the 'Deep Residual Learning for Image Recognition' paper?",
        "How many radiomic features were used in BRAINMETDETECT?",
        "What is the maximum depth of the trees used in deep probability aggregation clustering?",
        "How many federated agents were used in the FedMRL study?",
        "What is the learning rate used in periodic agent-state based Q-learning?",
        "How many epochs were used to train the 'Attention is All You Need' model?",
        "What is the input image size for the ResNet model?",
        "What specific radiomic features were most predictive in BRAINMETDETECT?",
        "How many hidden layers are in the analytic convolutional neural network?",
        "What are the hyperparameters for the deep probability aggregation clustering algorithm?",
        "What type of reinforcement learning algorithm is used in graph anomaly detection?",
        "How many agents are involved in the FedMRL medical imaging study?",
        "What is the training dataset size for the spacecraft thermal simulator model?",
        "How is the Q-learning update performed in periodic agent-state based Q-learning?",
        "What is the batch size used in training the personalized federated domain-incremental learning model?",
        "What is the size of the token vocabulary used in the 'Attention is All You Need' model?",
        "How many training samples were used in the 'Deep Residual Learning for Image Recognition' study?",
        "What is the accuracy of the BRAINMETDETECT model on the test set?",
        "What optimizer was used for training the deep probability aggregation clustering model?",
        "What is the dropout rate in the analytic convolutional neural network?",
        "How many features are used in the graph anomaly detection reinforcement learning model?",
        "What is the precision of the FedMRL model in medical imaging?",
        "What is the mean squared error of the physics-informed machine learning model for spacecraft thermal simulation?",
    ]
#28
    train_labels = [
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ]

    val_queries = [
        # General queries
        "Explain the architecture of a neural network.",
        "What are the key components of a convolutional layer?",
        "Summarize the contributions of the 'Deep Residual Learning for Image Recognition' paper.",
        "Describe the main ideas of reinforcement learning.",
        "What is the significance of using radiomic features in BRAINMETDETECT?",
        "Explain the concept of probability aggregation in clustering.",
        "How does federated learning apply to medical imaging?",
        "What are the challenges in graph anomaly detection?",
        "Describe the approach of periodic agent-state based Q-learning.",
        "What are the main features of personalized federated domain-incremental learning?",
        "How is machine learning used in spacecraft thermal simulation?",
        "What is the attention mechanism in transformers?",
        "Explain the purpose of residual connections in neural networks.",
        "What are the main applications of deep learning?",
        "Describe the architecture of ResNet.",
        "What is the impact of using reinforcement learning in graph anomaly detection?",
        "How does periodic agent-state based Q-learning work?",
        "What are the benefits of personalized federated learning?",
        "Explain the role of machine learning in spacecraft thermal simulation.",
        "What are the different types of neural networks?",
        "Describe the basics of machine learning.",
        "What is the main idea behind convolutional layers?",
        "What are the advantages of deep learning?",
        "Explain the concept of transformers in machine learning.",
        "What are the key findings of the 'Attention is All You Need' paper?",
        "Describe the methodology of the BRAINMETDETECT study.",
        "What is the role of reinforcement learning in graph anomaly detection?",
        "Explain the importance of periodic agent-state based Q-learning.",
        "What are the main contributions of personalized federated domain-incremental learning?",
        "Describe the applications of physics-informed machine learning in spacecraft simulation.",
        #30
        # Specific queries
        "What is the number of layers in ResNet-152?",
        "How many parameters does the 'Deep Residual Learning for Image Recognition' model have?",
        "What is the accuracy of the 'Attention is All You Need' model?",
        "What is the training dataset size for BRAINMETDETECT?",
        "What is the learning rate used in the deep probability aggregation clustering algorithm?",
        "How many epochs were used to train the FedMRL model?",
        "What is the precision of the graph anomaly detection model?",
        "How many hidden layers does the analytic convolutional neural network have?",
        "What is the batch size used in the periodic agent-state based Q-learning algorithm?",
        "How many agents were used in the FedMRL study?",
        "What is the maximum token length in the 'Attention is All You Need' model?",
        "What is the input image resolution for the ResNet model?",
        "How many radiomic features were used in BRAINMETDETECT?",
        "What is the test accuracy of the deep probability aggregation clustering model?",
        "What is the learning rate in the personalized federated domain-incremental learning model?",
        "What is the training duration for the physics-informed machine learning model?",
        "How many layers are in the transformer model?",
        "What is the dropout rate in the 'Attention is All You Need' model?",
        "How many training samples were used in the BRAINMETDETECT study?",
        "What optimizer was used for the deep probability aggregation clustering model?",
        "What is the accuracy of the graph anomaly detection model?",
        "How many epochs were used to train the periodic agent-state based Q-learning model?",
        "What is the mean squared error of the spacecraft thermal simulator model?",
        "What is the embedding size in the 'Attention is All You Need' model?",
        "What are the hyperparameters for the analytic convolutional neural network?",
    ]
#25
    val_labels = [
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    ]

    # max_len = 512
    batch_size = 8
    num_epochs = 10
    embedding_dim = 384  # For base model
    num_labels = 2  # General and specific

    train_dataset = QueryDataset(train_queries, train_labels)
    val_dataset = QueryDataset(val_queries, val_labels)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleClassifier(embedding_dim, num_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0025)
    loss_fn = nn.CrossEntropyLoss().to(device)


    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('------------------------')

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, len(train_dataset))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_dataset))
        print(f'Val loss {val_loss} accuracy {val_acc}')
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        save_model(model, f'models/model_epoch_{epoch + 1}.bin')


    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    print("Training complete.")

if __name__ == '__main__':
    main()
