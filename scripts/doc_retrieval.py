import json

# Function to load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Load the configuration
config = load_config('config.json')


search_endpoint = config["search_endpoint"]
search_api_key = config["search_api_key"] 
index_name = config["index_name"]

from azure.core.credentials import AzureKeyCredential

credential_search = AzureKeyCredential(search_api_key)

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

import embedder as embedder
from sklearn.metrics.pairwise import cosine_similarity
import torch
# Get search client

def doc_retrieval(query: str):
    search_service_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential_search)

    e1,_ = embedder.embedder(query, 128)

    # vector = embedder.embedder(query, max_length=128).squeeze().numpy().tolist()
    vector,_ = embedder.embedder(query, max_length=128)
    vector=vector.squeeze().tolist()
    print("HERE")
    
    vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=5, fields="embedding", exhaustive=False)
    results =  search_service_client.search(
        search_text=query,
        include_total_count=True,
        vector_queries=[vector_query]
    )
    # print("results: ", results)
    ret = []
    # print("number of docs: ", results.get_count())
    # print("query: ", query)
    # print("query embedding: ", vector[0:10])
    
    for result in results:
        # print("\n")
        # print("**********NEW DOCUMENT**********")
        # content = result["content"]
        # res_vector = result["embedding"]
        # e2 = torch.Tensor(res_vector)
        # print("vector type: ", type(e1))
        # print("azure embedding type: ", type(e2))
        # similarity_score = cosine_similarity(e1, e2.reshape(1,-1))
        # print("result content: ", content[0:20])
        # print("result embedding: ", res_vector[0:10])
        # print("cosine similarity: ", similarity_score[0][0])
        # print("result search score: ", result['@search.score'])
        # print("********END DOCUMENT*********")
        # print("\n")
        ret.append(result['content'])
        # if len(ret) == 4:
        #     break
            # print("results: ", results['content'])
    # ret = []
    # total_num_results = results.get_count()

    # print("documents found: ", total_num_results)
    
    # nums = 1
    # if total_num_results >= 100:
    #     nums = 20
    # elif total_num_results >= 10:
    #     nums = total_num_results // 5 + 1
    # elif total_num_results >= 8:
    #     nums = 3
    # elif total_num_results >= 5:
    #     nums = 2
 
    # print("nums: ", nums)
    # for result in results:
    #     if result["@search.score"] > 2.0:
    #         ret.append(result['content'])

    #     nums -= 1
    #     if nums == 0:
    #         break

    # return ret
    return ret    
  
        


def main():
    # query = "Provided proper attention"
    # thing = doc_retrieval(query=query)
    # for thin in thing:
    #     print("***********************")
    #     print(thin[0:20].lstrip())
    #     print("***********************")
    pass
if __name__ == '__main__':
    main()

