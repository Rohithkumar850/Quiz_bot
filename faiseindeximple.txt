import os
import fitz
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json
import faiss

def process_pdfs(folder_path, model_name, faiss_index_path):
    if not os.path.exists(folder_path):
        print(f"Source folder '{folder_path}' does not exist.")
        return

    # Get the current directory
    current_directory = os.getcwd()

    # Create a deposit folder in the current directory
    deposit_folder = os.path.join(current_directory, "Processed_Docs")
    os.makedirs(deposit_folder, exist_ok=True)

    # List PDF files in the source folder
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

    # Initialize the Hugging Face BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Initialize FAISS index
    d = 768  # Dimensionality of the embeddings, adjust based on the BERT model used
    index = faiss.IndexFlatL2(d)

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing PDF file: {pdf_file}")

        # Create a folder with the same name as the PDF file (without extension)
        folder_name = os.path.splitext(os.path.basename(pdf_file))[0]
        output_folder = os.path.join(deposit_folder, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Initialize PyMuPDF
        pdf_document = fitz.open(pdf_file)

        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text()

            # Split the extracted text into chunks (assuming a chunk size of 250 characters)
            chunk_size = 250
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            for i, chunk in enumerate(chunks):
                # Tokenize the text for each chunk
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)

                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling

                # Convert the PyTorch tensor to a NumPy array for FAISS
                embeddings_np = embeddings.numpy()

                # Add embeddings to FAISS index
                index.add(embeddings_np)

    # Save the FAISS index after processing all PDF files
    faiss.write_index(index, faiss_index_path)
                
                
                

if __name__ == "__main__":
    
    current_directory = os.getcwd()
    
    # Specify the source folder path containing PDF files
    source_folder_path = os.path.join(current_directory, "Docs")
    

    # Specify the Hugging Face BERT model name (e.g., 'bert-base-uncased')
    model_name = 'bert-base-uncased'

    # Specify the path to save the FAISS index
    faiss_index_path = os.path.join(current_directory, "faiss_index.index")

    # Process the PDF files, extract chunks, create embeddings, and build the vector database
    process_pdfs(source_folder_path, model_name, faiss_index_path)
