from sentence_transformers import SentenceTransformer
import faiss
import os
from gpt4all import GPT4All, Embed4All
import numpy as np
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import gpt4all


local_path = "C:/Users/Steve/AppData/Local/nomic.ai/GPT4All"
model_path = "C:\\Users\\Steve\\llama.cpp\\models\\ggml-vocab-falcon.gguf"

def load_text_file(file_path, encoding='utf-8'):
    """
    Load the contents of a text file.
    Parameters:
    - file_path (str): The path to the text file.
    Returns:
    - str: The content of the text file.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None

def split_into_chunks(text, chunk_size=512):
    """
    Split the text into chunks of a specified size.
    Parameters:
    - text (str): The input text.
    - chunk_size (int): The size of each chunk. Default is 512.
    Returns:
    - list: A list of chunks.
    """
    #return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return '\n'.join(chunks)

def create_embeddings(text_chunks):
    """
    Create embeddings for a list of text chunks.
    Parameters:
    - text_chunks (list): A list of text chunks.
    Returns:
    - numpy.ndarray: An array of embeddings.
    """
    
    model = Embed4All()
    #model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    #embeddings = model.encode(text_chunks)
    embeddings_vector = model.embed(text_chunks)
    print(f"\n\n\tTHIS IS THE LENTH OF THE EMBBEDINGS VECTOR: {len(embeddings_vector)}\n\n")
    print(embeddings_vector)
 
    
    return embeddings_vector

def index_embeddings(embeddings):
    """
    Index embeddings using FAISS.

    Parameters:
    - embeddings (numpy.ndarray): An array of embeddings.

    Returns:
    - faiss.IndexFlatL2: A FAISS index.
    """
    #embedding_size = len(embeddings[0])
    embedding_array = np.array(embeddings).reshape(1, -1)
    embedding_array.tolist()
    a = embedding_array.shape
    b = embedding_array.ndim
    c = embedding_array.size
  
    print(f"\n\n\t THIS IS GIVING THE \n\tSHAPE:{a}\n\tNUMBER OF DIMENSIONS:{b}\n\tSIZE OF THE EMEDDING ARRAY:{c}")
    
    #dimension = 256
    print(f"\n\n\tTHIS IS FROM THE INDEX_EMBEDDINGS \n\n{embedding_array}")
    
    dimension = len(embedding_array[0])
    print(f"\n\n\tDimension used for IndexFlatL2: {dimension}\n\n")
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array.astype('float32'))
    
    return index, embedding_array

# def search_similar_chunks(query_embedding, index, k=5000):
def search_similar_chunks(query_vector, index,embeddings, k=10):
    """
    Search for similar chunks given a query embedding.
    Parameters:
    - query_embedding (numpy.ndarray): The query embedding.
    - index (faiss.IndexFlatL2): The FAISS index.
    - k (int): The number of similar chunks to retrieve. Default is 5.
    Returns:
    - list: Indices of similar chunks.
    """
    
    query_array = np.array(query_vector).reshape(1, -1).astype('float32')
    print(f"Query vector dimensions: {query_array.shape}")
    print(f"Index dimensions: {index.d}")
    _, indices = index.search(query_array, k)
    # return indices.flatten()
    # Retrieve similar chunks
    similar_chunks = [embeddings[i] for i in indices.flatten()]
    
    return similar_chunks

if __name__ == "__main__":
    # Example usage:
    current_directory = os.getcwd()
    file_name = "MergedDocuments.txt"  # Replace 'example.txt' with the path to your text file
    file_path = os.path.join(current_directory, file_name)

    loaded_content = load_text_file(file_path)

    if loaded_content is not None:
        # Split the content into 512-character chunks
        chunks = split_into_chunks(loaded_content)
        # Create embeddings for the chunks
        embeddings = create_embeddings(chunks) # return the Query Vectors
        # Index the embeddings using FAISS
        index,Vector_arrays = index_embeddings(embeddings)
        # Example: Search for similar chunks given a query embedding
        
        # query_index = 300  # Replace with the index of the query chunk
        # query_embedding = embeddings[query_index]
        
        
        # Query_vector = embeddings[:]
        
        Query_vector = Vector_arrays[0]
        # Search for similar chunks
        similar_chunks = search_similar_chunks(Query_vector, index, embeddings)

        print(similar_chunks)
        