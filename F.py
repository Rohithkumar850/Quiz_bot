from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import gpt4all
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path 
from langchain.vectorstores import faiss
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os
from transformers import BertModel, BertTokenizer
from langchain.embeddings import OpenAIEmbeddings
import os
import getpass

# openai_api_key = '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'
# os.environ['OPENAI_API_KEY'] = getpass.getpass(openai_api_key)
#os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")
# current_directory = os.getcwd()
# file_name = "Documents.txt"  # Replace 'example.txt' with the path to your text file
# file_path = os.path.join(current_directory, file_name)
    
# loader = TextLoader(file_path,encoding='utf-8')
# #loader = PyPDFLoader(file_path=file_path)
# documents = loader.load_and_split()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
# texts = text_splitter.split_documents(documents)
# print(len(texts))

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# faiss_index = FAISS.from_documents(texts, embeddings)
# faiss_index_name = 'faiss-index-250'
# faiss_index.save_local(faiss_index_name)


current_directory = os.getcwd()
folder_name = "ConvertedDocs"  # Replace with the name of your folder containing PDF files
folder_path = os.path.join(current_directory, folder_name)

# Iterate through PDF files in the folder
pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

# Load and split PDF documents
documents = []
for pdf_file in pdf_files:
    pdf_file_path = os.path.join(folder_path, pdf_file)
    #pdf_loader = PyPDFLoader(pdf_file_path)
    pdf_loader = TextLoader(pdf_file_path,encoding='utf-8')
    documents.extend(pdf_loader.load_and_split())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(documents)
print(len(texts))

# Embed the text chunks using Hugging Face embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings =  OpenAIEmbeddings()

# Create a FAISS index from the embedded text chunks
faiss_index = FAISS.from_documents(texts, embeddings)

# Save the FAISS index locally
faiss_index_name = 'faiss-index-250'
faiss_index.save_local(faiss_index_name)