from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.vectorstores import faiss
from langchain.llms import gpt4all, llamacpp
from langchain.llms import GPT4All
from transformers import BertModel, BertTokenizer
import os
import argparse
import time


load_dotenv()

model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = os.environ.get('MODEL_N_BATCH')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS'))
    

def main():
    args = parse_arguments()
    FAISS_INDEX_PATH = './faiss-index-250'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    retriever = faiss_index.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    #llm = GPT4All(model_path=model_path, max_tokens=1000, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    #llm = GPT4All('ggml-model-gpt4all-falcon-q4_0.bin')
    # llm = GPT4All(model_path=model_path,model_type='gptj')
    llm = GPT4All(model='ggml-model-gpt4all-falcon-q4_0.bin', max_tokens=20, backend='gptj',n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents = not args.hide_source)
    # Questions and answers in Loop
    while True:
        query = input("\nEnter a Query (or type exit): ")
        if query == "exit":
            break
        if query.strip() =="":
            continue
        
        print("\nDEBUG: Input Question -", query)
        #get answers from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        
        print("\nDEBUG: Intermediate Results -", res)

        #print result
        print("\n\n> Question: ")
        print(query)
        #print(docs)
        print(f"\n> Answer (took {round(end - start, 2)}s.):")
        
        if docs:
            for i, doc in enumerate(docs, start=1):
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = metadata.get('source', 'N/A')
                content = doc.page_content if hasattr(doc, 'page_content') else 'Content not available'

                print(f"\nSource Document {i} - {source}:")
                print("Page Content:")
                print(content)

        # print("\nFinal Answer:")
        # print(answer)
        
        #print result
        # print("\n\n> Question: ")
        # print(query)
        # print(f"\n> Answer (took {round(end-start, 2)}s.):")
        # print(docs)
    

    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions on the document Privately, using the power of LLMs and gpt4All.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of Source Documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callbacks form LLMs.')
    return parser.parse_args()

if __name__ == "__main__":
    main()