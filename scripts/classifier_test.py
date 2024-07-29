import classifier, embedder
import torch

def test_model(model, test_queries, test_labels):
    model.eval()
    correct = 0
    total = len(test_queries)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, query in enumerate(test_queries):
            p,_ = classifier.predict(model, query, 128)
            # Compare with actual labels
            if p == test_labels[i]:
                correct += 1
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy


def main():

    test_queries = [
        # General queries
        "What is the purpose of data augmentation in machine learning?",
        "How does dropout prevent overfitting in neural networks?",
        "Summarize the key points of the BERT model.",
        "Describe the differences between LSTM and GRU.",
        "What are the main advantages of using convolutional layers in CNNs?",
        "Explain the concept of transfer learning.",
        "What is the role of attention mechanisms in NLP models?",
        "How does gradient descent work?",
        "What are the benefits of using ensemble methods?",
        "What is the significance of the Transformer model in NLP?",
        "Describe the architecture of a recurrent neural network.",
        "What is the impact of using batch normalization?",
        "Explain the concept of autoencoders.",
        "How is reinforcement learning different from supervised learning?",
        "What are the key contributions of the GPT model?",
        "Describe the main features of a decision tree.",
        "How does the skip-gram model work in word2vec?",
        "What is the purpose of using activation functions in neural networks?",
        "Explain the concept of backpropagation.",
        "What are the main challenges in training deep neural networks?",
        "What is a variational autoencoder?",
        "Describe the architecture of a GAN.",
        "What is the purpose of using attention in sequence-to-sequence models?",
        "How does the Adam optimizer work?",
        "What are the differences between precision and recall?",
        "Explain the concept of hyperparameter tuning.",
        "What is the purpose of using embeddings in NLP?",
        "Describe the key features of the YOLO object detection model.",
        "What are the benefits of using a pre-trained model?",
        "Explain the concept of clustering in unsupervised learning.",
        "What is the role of the encoder in a transformer model?",
        "How does data normalization improve model performance?",
        "What are the main applications of RNNs?",
        "Describe the architecture of the AlexNet model.",
        "What are the advantages of using LSTM over traditional RNNs?",
        "How does the beam search algorithm work?",
        "What is the purpose of using an attention mask in transformers?",
        "Explain the impact of using normalization techniques in machine learning.",
        "What are the challenges in using GANs for image generation?",
        "Describe the transfer learning process for computer vision tasks.",
        "What is the impact of using different activation functions?",
        
        # Specific queries
        "How many heads are used in the multi-head attention mechanism of BERT?",
        "What is the learning rate used in the original GPT model?",
        "How many parameters does the YOLOv3 model have?",
        "What is the image input size for the AlexNet model?",
        "How many epochs were used to train the BERT model?",
        "What is the batch size used in training GPT-3?",
        "How many hidden units are in the LSTM layer of the original LSTM paper?",
        "What is the precision of the YOLOv3 model on the COCO dataset?",
        "What optimizer was used for training the original AlexNet model?",
        "What is the dropout rate in the BERT base model?",
        "How many layers are there in the Transformer model?",
        "What is the embedding size in the GPT-2 model?",
        "How many training samples were used in the BERT large model?",
        "What is the test accuracy of the YOLOv4 model?",
        "What is the learning rate in the Adam optimizer used in GPT-3?",
        "How many epochs were used to train the LSTM model?",
        "What is the mean squared error of the VAE model?",
        "How many layers are in the BERT base model?",
        "What is the dropout rate in the GPT-2 model?",
        "What are the hyperparameters for the YOLOv3 model?",
        "What is the batch size used in training the Transformer model?",
        "How many epochs were used to train the AlexNet model?",
        "What is the accuracy of the LSTM model on the IMDB dataset?",
        "What optimizer was used for training the GPT-2 model?",
        "How many parameters does the original LSTM model have?",
        "What is the input image size for the YOLOv3 model?",
        "What is the precision of the BERT model on the SQuAD dataset?",
        "What is the learning rate in the Adam optimizer used in GPT-2?",
        "What are the key hyperparameters in training the GAN model?",
        "How many filters does the first convolutional layer in AlexNet have?",
        "What is the context size used in the skip-gram model?",
    ]


    test_labels = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # Specific queries
    ]


    model = classifier.SimpleClassifier(768, 2) # 768 embed_dim, 2 labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model = classifier.load_model(model, 'models/model_epoch_10.bin', device)

    accuracy = test_model(model, test_queries, test_labels)
    print("Accuracy: ", accuracy)
    # # print(len(test_queries))
    # print(len(test_labels))

if __name__ == "__main__": main()