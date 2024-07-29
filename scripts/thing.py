# import os
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# import doc_retrieval, embedder, classifier
# import time
# max_length_long = 512  
# max_length_short = 128  

# # Function to load existing embeddings
# def load_embeddings(file_path):
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#     else:
#         return {}

# # Function to save embeddings
# def save_embeddings(embeddings, file_path):
#     with open(file_path, 'wb') as f:
#         pickle.dump(embeddings, f)


# # Function to conduct semantic search
# def semantic_search(query_embedding, embeddings):
#     similarities = {}
#     for doc_name, doc_embedding in embeddings.items():
#         # #print("query_embedding: ", query_embedding.dim())
#         # #print("doc_embedding: ", doc_embedding.dim())
#         similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
#         similarities[doc_name] = similarity
#     sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
#     return sorted_similarities

# # Function to classify the question
# def classify_question(query):
#     # Placeholder classifier, replace with actual implementation
#     # For example, use a pre-trained classifier model
    
#     model = classifier.SimpleClassifier(768, 2) # 768 embed_dim, 2 labels

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = classifier.load_model(model, 'models/model_epoch_10.bin', device)
#     model.eval()
#     # #print(type(query_embedding))
#     predicted_class, _ = classifier.predict(model, query, max_length_short)
#     # #print("predicted: ", predicted_class)
#     if predicted_class == 0: return "General."
#     elif predicted_class == 1 : return "Specific."




# # Function to answer the question
# def answer_question(query):
#     long_chunk_embeddings = load_embeddings('long_chunk_embeddings.pkl')
#     short_chunk_embeddings = load_embeddings('short_chunk_embeddings.pkl')

#     # Retrieve documents using your document_retrieval function
#     retrieved_docs = doc_retrieval.doc_retrieval(query)
    
#     # Embed documents only if not already embedded
#     new_long_chunk_embeddings = {}
#     new_short_chunk_embeddings = {}

#     for doc_text in retrieved_docs:
#         doc_name = f'doc_{doc_text[0:20].lstrip()}' ## this doesnt work because doc_i will change for every search
#         if doc_name not in long_chunk_embeddings:
#             new_long_chunk_embeddings[doc_name] = embedder.embedder(doc_text, max_length_long)
#         if doc_name not in short_chunk_embeddings:
#             new_short_chunk_embeddings[doc_name] = embedder.embedder(doc_text, max_length_short)

#     # Update the embeddings with new ones
#     long_chunk_embeddings.update(new_long_chunk_embeddings)
#     short_chunk_embeddings.update(new_short_chunk_embeddings)

#     # Save the updated embeddings
#     save_embeddings(long_chunk_embeddings, 'long_chunk_embeddings.pkl')
#     save_embeddings(short_chunk_embeddings, 'short_chunk_embeddings.pkl')

#     query_embedding = embedder.embedder(query, max_length=max_length_short)

#     long_chunk_results = semantic_search(query_embedding, long_chunk_embeddings)

#     question_type = classify_question(query)

#     if question_type == "General.":
#         return long_chunk_results[0][0]  # Return the most relevant long chunk
#     else:
#         short_chunk_results = []
#         for doc_name, _ in long_chunk_results:
#             short_chunk_results.extend(semantic_search(query_embedding, {doc_name: short_chunk_embeddings[doc_name]}))
#         return short_chunk_results[0][0]  # Return the most relevant short chunk

# # Example usage
# def main():
#     # Load existing embeddings
    
    
#     query = "What is attention"
#     # text = doc_retrieval.doc_retrieval(query=query)
#     # query_type = classify_question(query)
#     # #print("query_type: ", query_type)
#     # start_time = time.time()
#     answer = answer_question(query)
#     # end_time = time.time()
#     #print(f"Answer: {answer}")
#     # #print("total time: ", end_time - start_time)


# if __name__ == "__main__": main()













import os
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time

import doc_retrieval, classifier
import embedder as embedder

max_length_long = 512
max_length_short = 128

# Function to load existing embeddings
def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

# Function to save embeddings
def save_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Function to conduct semantic search
def semantic_search(query_embedding, embeddings):
    similarities = {}
    for doc_name, doc_embedding in embeddings.items():
        similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
        similarities[doc_name] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities

# Function to classify the question
def classify_question(model, device, query):
    model.eval()
    predicted_class, _ = classifier.predict(model, query, max_length_short)
    if predicted_class == 0:
        return "General."
    elif predicted_class == 1:
        return "Specific."

# Function to answer the question
def answer_question(query, model, device, long_chunk_embeddings, short_chunk_embeddings, long_chunk_texts, short_chunk_texts):
    #print("entered answer_question!")
    # Retrieve documents using your document_retrieval function
    #print("retrieving documents...")
    retrieved_docs = doc_retrieval.doc_retrieval(query)
    #print("documents retrieved!")
    
    # Embed documents only if not already embedded
    new_long_chunk_embeddings = {}
    new_short_chunk_embeddings = {}
    new_long_chunk_texts = {}
    new_short_chunk_texts = {}


    #print("loading chunk embeddings...")
    for doc_text in retrieved_docs:
        doc_name = f'doc_{doc_text[0:20].lstrip()}'  # this doesn't work because doc_i will change for every search
        if doc_name not in long_chunk_embeddings:
            long_e, long_c = embedder.embedder(doc_text, max_length_long)
            new_long_chunk_embeddings[doc_name] = long_e
            new_long_chunk_texts[doc_name] = long_c
        if doc_name not in short_chunk_embeddings:
            short_e, short_c = embedder.embedder(doc_text, max_length_short)
            new_short_chunk_embeddings[doc_name] = short_e
            new_short_chunk_texts[doc_name] = short_c

    #print("chunks loaded!")

    # Update the embeddings with new ones
    
    long_chunk_embeddings.update(new_long_chunk_embeddings)
    short_chunk_embeddings.update(new_short_chunk_embeddings)
    long_chunk_texts.update(new_long_chunk_texts)
    short_chunk_texts.update(new_short_chunk_texts)
    #print("chunk embeddings loaded!")

    # Save the updated embeddings
    #print("saving embeddings...")
    save_embeddings(long_chunk_embeddings, 'long_chunk_embeddings.pkl')
    save_embeddings(short_chunk_embeddings, 'short_chunk_embeddings.pkl')
    save_embeddings(long_chunk_texts, 'long_chunk_texts.pkl')
    save_embeddings(short_chunk_texts, 'short_chunk_texts.pkl')
    #print("embeddings saved!")

    #print("query being embedded...")
    query_embedding,_ = embedder.embedder(query, max_length=max_length_short)
    #print("query embedded!")

    #print("executing semantic search...")
    long_chunk_results = semantic_search(query_embedding, long_chunk_embeddings)
    #print("semantic search executed!")

    #print("classifying question...")
    question_type = classify_question(model, device, query)
    #print("question classified!")

    #print("inputting correct chunk...")
    ret = None
    doc= None
    if question_type == "General.":
        print("general")
        #print("question found to be general: inputting long chunk result.")
        #print("long_chunk_results: ", long_chunk_results)
        doc = long_chunk_results[0][0]  # Return the most relevant long chunk
        ret = long_chunk_texts[doc]
        #print("correct chunk filled!")
    else:
        #print("question found to be specific: executing semantic search on short chunks.")
        short_chunk_results = []
        print("specific")
        for doc_name, _ in long_chunk_results:
            #print("executing semantic search...")
            search = semantic_search(query_embedding, {doc_name: short_chunk_embeddings[doc_name]})
            #print("semantic search executed!")
            short_chunk_results.extend(search)

        # doc_name, _ = long_chunk_results[0][0]
        # #print("executing semantic search...")
        # search = semantic_search(query_embedding, {doc_name: short_chunk_embeddings[doc_name]})
        # #print("semantic search executed!")
        # short_chunk_results.extend(search)
        #print("correct chunk filled!")
        ##print("short_chunk_results: ", short_chunk_results)
        doc = short_chunk_results[0][0] # Return the most relevant short chunk
        ret = short_chunk_texts[doc]
    ##print("exiting answer_question")
    return doc, ret

def main():
    #print("*****START*****")
    # Load existing embeddings
    #print("loading embeddings...")
    long_chunk_embeddings = load_embeddings('long_chunk_embeddings.pkl')
    short_chunk_embeddings = load_embeddings('short_chunk_embeddings.pkl')
    long_chunk_texts = load_embeddings('long_chunk_texts.pkl')
    short_chunk_texts = load_embeddings('short_chunk_texts.pkl')
    #print("loaded embeddings!")
    # Load classifier model
    #print("created model")
    model = classifier.SimpleClassifier(384, 2)  # 768 embed_dim, 2 labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("loading model...")
    model = classifier.load_model(model, 'models/model_epoch_10.bin', device)
    #print("loaded model!")

    query = "attention"
    #print("answering question...")
    start_time = time.time()
    answer = answer_question(query, model, device, long_chunk_embeddings, short_chunk_embeddings, long_chunk_texts, short_chunk_texts)
    end_time = time.time()
    
    print(f"Done! Answer: {answer}")
    print("Total time: ", end_time - start_time)
    #print("*****END******")

if __name__ == "__main__":
    main()
