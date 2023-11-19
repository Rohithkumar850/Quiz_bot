import os
import fitz
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import json

def process_pdfs(folder_path, model_name):
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

            # Split the extracted text into chunks (assuming a chunk size of 1000 characters)
            chunk_size = 1000
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

                # Read the content of the chunk file
                with open(chunk_filepath, 'r', encoding='utf-8') as chunk_file:
                    chunk_content = chunk_file.read()

                # The rest of your processing code...
                # Tokenize the text, get embeddings, save as JSON, etc.

if __name__ == "__main__":
    # Specify the source folder path containing PDF files
    source_folder_path = "D:\\Destroy_this\\Docs"

    # Specify the Hugging Face BERT model name (e.g., 'bert-base-uncased')
    model_name = 'bert-base-uncased'

    # Process the PDF files, extract chunks, create embeddings, and build the vector database
    process_pdfs(source_folder_path, model_name)
