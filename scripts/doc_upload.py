import os
import fitz  # PyMuPDF
from azure.storage.blob import BlobServiceClient

# Function to upload a file to Azure Blob Storage
def upload_to_blob_storage(blob_service_client, container_name, file_path, file_name):
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except Exception:
        pass  # Container already exists

    blob_client = container_client.get_blob_client(file_name)

    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file_name} to {container_name} container.")

# def chunk_text(text, max_length=512):
#     chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
#     return chunks

def chunk_text(text):
    max_length = len(text) // 2
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    return chunks

def read_pdf(file_path):
    doc = fitz.open(file_path)
    # print("doc: ", doc)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # print("page.get_text: ", page.get_text())
        text += page.get_text()
    return text


def main():
    # Input PDF document paths
    document_paths = []
    path = ""
    while path != "*stop*":
        path = input("Enter document path (or type *stop* to finish): ")
        if path != "*stop*":
            document_paths.append(path)

    # Initialize Blob Service Client
    connect_str = "DefaultEndpointsProtocol=https;AccountName=teststorage6262024;AccountKey=DIokN/+/h5VvByCS6TZiOJoTXxeQvDZj5KK/IKFMDrKKG7+W5m8zsw5i4k4tL6FTN46NuK+hBeQ9+ASt3OMLQQ==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Container name for chunked documents
    container_name = "chunked-documents-2"

    # Chunk documents and upload to Blob Storage
    for doc_path in document_paths:
        doc_name = os.path.basename(doc_path).split('.')[0]
        doc_text = read_pdf(doc_path)
        # print(doc_text)
        chunks = chunk_text(doc_text)
        for i, chunk in enumerate(chunks):
            file_name = f"{doc_name}_chunk_{i}.txt"
            with open(file_name, "w") as f:
                f.write(chunk)
            upload_to_blob_storage(blob_service_client, container_name, file_name, file_name)
            os.remove(file_name)
    print("done")
if __name__ == "__main__":
    main()
