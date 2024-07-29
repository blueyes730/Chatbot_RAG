import torch
import numpy
from transformers import BertTokenizerFast, BertModel
from sklearn.preprocessing import normalize
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states= True)
# max_length = 512

model.eval()

def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    attention_mask = attention_mask.unsqueeze(-1)
    return (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

def list_to_string(text_list: list[str]):
    string = ""
    for text in text_list:
        string += text + " "
    
    return string[:-1]


def embedder(text, max_length=512):

    if isinstance(text, list):
        text = list_to_string(text)
    
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    
    all_hidden_states = []

    for chunk in chunks:

        tokenized_chunk = tokenizer(chunk, return_tensors='pt', truncation=True, padding='longest', max_length=max_length)

        input_ids = tokenized_chunk['input_ids']
        token_type_ids = tokenized_chunk['token_type_ids']
        attention_mask = tokenized_chunk['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            chunk_embedding = mean_pooling(outputs, attention_mask)
            all_hidden_states.append(chunk_embedding) # 

    embedding = torch.mean(torch.stack(all_hidden_states), dim=0)
    # Normalize the embeddings
    normalized_embedding = normalize(embedding.cpu().numpy(), axis=1)

    return normalized_embedding


def main():
    text = "GeeksforGeeks is a computer science portal"
    tokenized = embedder(text)
    print(tokenized)
    pass

if __name__ == '__main__':
    main()










