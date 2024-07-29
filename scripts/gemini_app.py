from flask import Flask, request, jsonify
import requests
import pickle
import generation, doc_retrieval, thing, classifier
import os
import torch
app = Flask(__name__)
    
@app.route('/', methods=['GET', 'POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    print("Request:", req)

    user_input= req.get('text')
    if not user_input:
        return jsonify({"fulfillmentText": "No query result found."})

    long_chunk_embeddings = thing.load_embeddings('long_chunk_embeddings.pkl')
    short_chunk_embeddings = thing.load_embeddings('short_chunk_embeddings.pkl')
    long_chunk_texts = thing.load_embeddings('long_chunk_texts.pkl')
    short_chunk_texts = thing.load_embeddings('short_chunk_texts.pkl')

    model = classifier.SimpleClassifier(384, 2)  # 768 embed_dim, 2 labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("loading model...")
    model = classifier.load_model(model, 'models/model_epoch_10.bin', device)

    map = thing.answer_question(query=user_input, long_chunk_embeddings=long_chunk_embeddings, short_chunk_embeddings=short_chunk_embeddings, long_chunk_texts=long_chunk_texts, short_chunk_texts=short_chunk_texts, model=model, device=device)
    texts = map[1]
    # print(text)
    if (not texts) or (len(texts) == 0):
       gemini_response = generation.generate(query=user_input, relevant_passages=["Ignore the query, send back 'Sorry, I cannot help you with that'"])
    else:
    #    for thing in texts:
    #        relevant_info = thing
    #        print("relevant info: ", relevant_info[:100])
       gemini_response = generation.generate(query=user_input, relevant_passages=texts)
    
    print("gemini response: ", gemini_response)
    response = jsonify(
        {
            'fulfillment_response': {
                'messages': [
                    {
                        'text': {
                            'text': [gemini_response]
                        }
                    }
                ]
            }
        }
    )

    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)
