from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def list_to_string(text_list: list[str]):
    string = ""
    for text in text_list:
        string += text + " "
    
    return string[:-1]

def embedder(text, max_length):
    if isinstance(text, list):
        text = list_to_string(text)
    
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    all_hidden_states = []
    chunk_texts = []

    for chunk in chunks:
        encoded_input = tokenizer(chunk, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        with torch.no_grad():
            model_output = model(**encoded_input)
        chunk_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        all_hidden_states.append(chunk_embedding)
        chunk_texts.append(chunk)

    embedding = torch.mean(torch.stack(all_hidden_states), dim=0)
    normalized_embedding = normalize(embedding.cpu().numpy(), axis=1)

    return normalized_embedding, chunk_texts


def main():
    # text = "GeeksforGeeks is a computer science portal"
    # tokenized = embedder(text)
    # print(tokenized.squeeze(0).shape)
    pass

if __name__ == "__main__": 
    main()
