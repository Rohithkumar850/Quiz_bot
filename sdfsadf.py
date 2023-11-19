import os
import fitz
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json

def process_pdfs(folder_path, model_name, vector_db_path):
    if not os.path.exists(folder_path):
        print(f"Source folder '{folder_path}' does not exist.")
        return

    current_directory = os.getcwd()

    # Create a deposit folder in the current directory
    deposit_folder = os.path.join(current_directory, "Processed_Docs")
    trail_path = os.path.join(current_directory, "Vector_DB")
    os.makedirs(deposit_folder, exist_ok=True)
    os.makedirs(trail_path, exist_ok=True)

    # List PDF files in the source folder
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]

    # Initialize the Hugging Face BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Initialize an empty dictionary to store embeddings
    vector_db = {}

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
                chunk_filename = f"page_{page_number + 1}_chunk_{i + 1}.txt"
                chunk_filepath = os.path.join(output_folder, chunk_filename)

                # Create the chunk file if it doesn't exist
                if not os.path.exists(chunk_filepath):
                    with open(chunk_filepath, 'w', encoding='utf-8') as new_chunk_file:
                        new_chunk_file.write(chunk)
                else:
                    print(f"Warning: File {chunk_filepath} already exists. Skipping creation.")

                # Tokenize the text
                tokens = tokenizer(chunk, return_tensors='pt')

                # Get the model embeddings
                with torch.no_grad():
                    outputs = model(**tokens)

                embeddings = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling

                # Convert the PyTorch tensor to a NumPy array for JSON serialization
                embeddings_np = embeddings.numpy()

                # Save the embeddings as a JSON file
                embeddings_filename = f"page_{page_number + 1}_chunk_{i + 1}_embeddings.json"
                embeddings_filepath = os.path.join(output_folder, embeddings_filename)

                with open(embeddings_filepath, 'w', encoding='utf-8') as json_file:
                    json.dump({'embeddings': embeddings_np.tolist()}, json_file)

                # Store the embeddings file path in the vector database
                vector_db[os.path.basename(embeddings_filepath)] = embeddings_filepath

                print(f"Saved embeddings for chunk {i + 1} from page {page_number + 1} to: {embeddings_filepath}")


    # Save the vector database
    vector_db_filename = os.path.join(vector_db_path, "vector_database.json")
    
    # Ensure the directory exists before writing the file
    os.makedirs(os.path.dirname(vector_db_filename), exist_ok=True)
    try:
        with open(vector_db_filename, 'w', encoding='utf-8') as vector_db_file:
            json.dump(vector_db, vector_db_file)
    except Exception as e:
        print(f"Error writing vector database to {vector_db_filename}: {e}")

    print(f"Vector database saved to: {vector_db_filename}")

if __name__ == "__main__":
    
    current_directory = os.getcwd()
    
    # Specify the source folder path containing PDF files
    source_folder_path = os.path.join(current_directory, "Docs")

    # Specify the Hugging Face BERT model name (e.g., 'bert-base-uncased')
    model_name = 'bert-base-uncased'

    # Specify the path for the vector database
    vector_db_path = os.path.join(current_directory, "Vector_DB")

    # Process the PDF files, extract chunks, create embeddings, and build the vector database
    process_pdfs(source_folder_path, model_name, vector_db_path)
