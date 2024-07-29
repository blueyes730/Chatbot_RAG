import pathlib
import textwrap
import os
import doc_retrieval, thing, classifier
import torch
import google.generativeai as genai

import json

# Function to load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Load the configuration
config = load_config('config.json')

def make_rag_prompt(query: str, relevant_passages: list[str]):
  # print("relevant passage: ", relevant_passage)
  num = 1
  relevant_info = ""
  for relevant_passage in relevant_passages:
    # print("current length: ", len(relevant_info))
    edited = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    # print("length adding: ", len(edited))
    # print("info being added: ", edited[:100])
    
    relevant_info += ("document '{num}': '{edited}'").format(num=num, edited=edited)
    num += 1

  # prompt = ("""
  # You are a helpful and informative bot that answers questions using text from all of the reference documents included below in the form of one giant string in this format: 'document: passage'. \
  # Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
  # However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
  # strike a friendly and conversational tone without letting the audience know you are referring to passages. 
  # You are allowed to greet people back and answer friendly questions such as 'How are you?' or 'What's up'.
  # Information used to generate the answer must be from the supplied documents only. 
  # QUESTION: '{query}'
  # PASSAGE: '{relevant_passage}'
  # ANSWER:
  # """).format(query=query, relevant_passage=relevant_info)

  prompt = ("""
  You are a helpful and informative bot that answers questions using text from all of the reference documents included below in the form of one giant string in this format: 'document: passage'. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
  strike a friendly and conversational tone without letting the audience know you are referring to passages. 
  You are allowed to greet people back and answer friendly questions such as 'How are you?' or 'What's up'.
  If the passages are irrelevant, just return 'I cannot answer that, sorry."
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'
  ANSWER:
  """).format(query=query, relevant_passage=relevant_info)

  return prompt

def generate(query: str, relevant_passages: list[str]):
    GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
    gemini_api_key = GOOGLE_API_KEY
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    # model = genai.GenerativeModel('gemini-pro')
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = make_rag_prompt(query=query, relevant_passages=relevant_passages)
    # print("prompt: ", prompt)
    answer = model.generate_content(prompt)
    return answer.text

def main():
    # while (1):
    #     query = input("Enter query: ")
    #     if (query == "End chat."):
    #        break
    #     text = doc_retrieval.doc_retrieval(query=query)
    #     # print(f"content: {doc['content']}\n")
        
    #     answer = generate(query=query, relevant_info=text["content"])
    #     print("ANSWER: ", answer)
    query = "what role does self-attention play transformers"
    long_chunk_embeddings = thing.load_embeddings('long_chunk_embeddings.pkl')
    short_chunk_embeddings = thing.load_embeddings('short_chunk_embeddings.pkl')
    long_chunk_texts = thing.load_embeddings('long_chunk_texts.pkl')
    short_chunk_texts = thing.load_embeddings('short_chunk_texts.pkl')

    model = classifier.SimpleClassifier(384, 2)  # 768 embed_dim, 2 labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("loading model...")
    model = classifier.load_model(model, 'models/model_epoch_10.bin', device)

    map = thing.answer_question(query=query, long_chunk_embeddings=long_chunk_embeddings, short_chunk_embeddings=short_chunk_embeddings, long_chunk_texts=long_chunk_texts, short_chunk_texts=short_chunk_texts, model=model, device=device)
    texts = map[1]
    # print(f"content: {doc['content']}\n")
    if not texts:
       answer = generate(query=query, relevant_passages=["Ignore the query, send back 'Sorry, I cannot help you with that'"])
    else:
       answer = generate(query=query, relevant_passages=texts)
    print("ANSWER: ", answer)
    pass

if __name__ == '__main__':
    main()
