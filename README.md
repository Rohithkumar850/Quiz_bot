# Quiz_bot
Creating a quiz bot

**Project Description**: 

Develop a local quiz bot focused on network security, capable of generating random questions and offering specific topic questions. The bot will utilize open-source alternatives to ChatGPT to run locally, ensuring data privacy by keeping information within the user's system. The quiz will include multiple-choice questions (MCQs), true/false questions, and open-ended questions, with answers sourced from a network security database.

**System Architecture**:

![LLM](https://github.com/Rohithkumar850/Quiz_bot/assets/68658502/3808af3e-8bb1-4ba4-b84f-96372b3bb000)



1.In the user interface, the user types a prompt. 
2. The application creates an embedding from the user's prompt and sends it to the vector database using the embedding model. 
3. Based on how closely the documents' embeddings resemble the user's prompt, the vector database provides a list of pertinent documents. 4. After retrieving the documents, the application constructs a new prompt and sends it to the local LLM using the user's original prompt as the context. 
5. The output is generated by the LLM, which also includes citations from the relevant texts. The user interface shows the outcome.

**Prerequisite**:

**1. Choosing a Programming Language:**
For developing projects which involve natural language processing and chatbot development Python is widely used. Thus, we have chosen Python as our programming language for development of the quiz bot

**2. Large Language Model:**
Rather than building the quizbot from scratch we have chosen to use gpt4all as the base on used falcon as the base on build upon it 

**3. Collecting Data:**
We have collected data from the network security slides to formulate the questions.

**System Requirements**
-Operating System:
Windows 10 /11 
RAM: 12GB
Disk Space Required:4 GB

**Dependencies/Packages:**
**-Pytorch**: PyTorch is a machine learning framework based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI and now part of the Linux Foundation umbrella. It is free and open-source software released under the modified BSD license
**-Faiss**: Faiss is a library — developed by Facebook AI — that enables efficient similarity search. So, given a set of vectors, we can index them using Faiss — then using another vector (the query vector), we search for the most similar vectors within the index.
**-Astra DB Vector**: Astra DB gives you elegant APIs, powerful data pipelines and complete ecosystem integrations to quickly build Gen AI applications on your real-time data for more accurate AI that you can deploy in production.
**Langchain**: LangChain is a framework designed to simplify the creation of applications using large language models. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis
**Alpaca-Native-7B-ggml:**We introduce Alpaca 7B, a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations. On our preliminary evaluation of single-turn instruction following, Alpaca behaves qualitatively similarly to OpenAI's text-davinci-003, while being surprisingly small and easy/cheap to reproduce

**Execution Instructions:**
1. load the GPT4All model
2. use Langchain to retrieve our documents and Load them
3. split the documents in small chunks digestible by Embeddings
4. Use FAISS to create our vector database with the embeddings
5. Perform a similarity search (semantic search) on our vector database based on the question we want to pass to GPT4All: this will be used as a context for our question
6. Feed the question and the context to GPT4All with Langchain and wait for the the answer.
   
**Features**:

- **Large-scale language model** - It utilizes a large language model architecture similar to GPT-3 and GPT-4, allowing it to generate human-like text and have conversations.

- **Local execution** - Models can be run locally without relying on APIs or cloud providers. This allows greater control, customization, and privacy.

- **Modular design** - The codebase is designed to be modular and extensible, making it easy for developers to build custom solutions.

- **Multiple model sizes** - Smaller models for basic use cases as well as larger models for more advanced capabilities are supported.

- **Built for safety** - gpt4all incorporates techniques like Constitutional AI to improve model safety and mitigate harmful behavior.

- **Active learning** - Models can be further trained through active learning to continuously improve and customize them for specific domains.

- **Easy integration** - REST APIs, embeddable widgets, and Python libraries allow integrating gpt4all into applications.

- **Comprehensive docs** - Detailed documentation covers training, evaluating, deploying, and using gpt4all models.

- **Community-driven** - The open source community collaboratively builds, supports, and enhances gpt4all.

**Training Data**
The data to be trained is extracted from the lectures using the embedded model and formulated into questions of multiple choice, true/false and open ended questions

**Data Formats**
**-.txt
-.pdf
-.docx**

