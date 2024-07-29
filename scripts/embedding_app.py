from flask import Flask, request, jsonify
import requests

import embedder as embedder

app = Flask(__name__)

@app.route('/', methods=['POST'])
def vectorize():
    req = request.get_json(silent=True, force=True)
    print("Request JSON:", req)
    values = req.get('values')
    if not values:
        return jsonify({"error": "No values found in the request."})

    response_values = []
    for record in values:
        record_id = record.get('recordId')
        data = record.get('data')
        text = data.get('text')

        if not text:
            return jsonify({"error": f"No text found for recordId {record_id}."})

        embedding,_ = embedder.embedder(text).squeeze().tolist()
        
        response_values.append({
            "recordId": record_id,
            "data": {
                "embedding": embedding
            }
        })

    response = {
        "values": response_values
    }

    return jsonify(response)
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
