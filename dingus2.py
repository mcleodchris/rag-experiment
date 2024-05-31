import argparse
import glob
import os
import pickle
import re

import numpy as np
import torch
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv
from openai import OpenAI
import ollama
import time
'''
 This version of Dingus (v2) is a smooshing together of ideas from Dingus v1,
 and code by All About AI:
 - Video: https://www.youtube.com/watch?v=Oe-7dGDyzPM
 - Repo: https://github.com/AllAboutAI-YT/easy-local-rag

'''

# Load environment variables from .env file
load_dotenv()

DOCUMENTS_FILE = os.environ.get('DOCUMENTS_FILE',"documents2.pkl")
EMBEDDINGS_FILE = os.environ.get('EMBEDDINGS_FILE',"embeddings2.npy")
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL','phi3:mini')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL','mxbai-embed-large')
LOCAL_DOCUMENTS = os.environ.get('DOCS_LOCATION', '')
OPENAI_API_BASE = os.environ.get('OPENAI_BASE_URL', 'http://localhost:11434/v1')

client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OLLAMA_MODEL
)

# File extraction functions
def extract_text_from_html(file_path):
    """
    Extracts text content from an HTML file.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        str: The extracted text content from the HTML file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text(strip=True)

def extract_text_from_docx(file_path):
    """
    Extracts the text content from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text content.
    """
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def extract_text_from_txt(file_path):
    """
    Extracts the text content from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.

    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    """
    Extracts text from a given file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text from the file.

    Raises:
        None

    """
    extractors = {
        ".html": extract_text_from_html,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".md": extract_text_from_txt
    }
    ext = os.path.splitext(file_path)[1].lower()
    extractor = extractors.get(ext)
    if extractor:
        return extractor(file_path)


def chunk_text(text):
    """
    Split a given text into chunks by sentences, respecting a maximum chunk size.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of chunks, where each chunk is a string of sentences separated by two newlines.
    """
    # Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    # Split text into chunks by sentences, respecting a maximum chunk size
    sentences = re.split(r'(?<=[.!?]) +', text)  # split on spaces following sentence-ending punctuation
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check if the current sentence plus the current chunk exceeds the limit
        if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
            current_chunk += (sentence + " ").strip()
        else:
            # When the chunk exceeds 1000 characters, store it and start a new one
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:  # Don't forget the last chunk!
        chunks.append(current_chunk)
    # return a string of all chunks separated by 2 newlines
    return chunks

def load_documents():
    """
    Load documents from disk or pickle file.

    If the pickle file exists, it loads the documents from the file.
    Otherwise, it loads the documents from the disk by extracting text from the specified file paths,
    chunking the text, and appending the chunks to the documents list.

    Returns:
        List: The list of loaded document fragments.
    """
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, 'rb') as f:
            documents = pickle.load(f)
    else:
        documents = []
    # if no documents, load from disk
    if not documents:
        print("No pre-processed documents found. Loading from disk...")
        file_paths = []
        if ',' in LOCAL_DOCUMENTS:
            segments = LOCAL_DOCUMENTS.split(',')
            for segment in segments:
                file_paths.extend(glob.glob(segment.strip(), recursive=True))
        else:
            file_paths = glob.glob(LOCAL_DOCUMENTS, recursive=True)
        for file_path in file_paths:
            try:
                text = extract_text(file_path)
                if text:
                    chunks = chunk_text(text)
                    # append all chunks to documents
                    documents.extend(chunks)
            except ValueError as e:
                print(e)
        print(f"Processed {len(documents)} document fragments.")
        #pickle documents
        with open(DOCUMENTS_FILE, 'wb') as f:
            print("Pickling documents...")
            pickle.dump(documents, f)
    return documents

def generate_embeddings(documents):
    """
    Generate embeddings for a list of documents.

    If the embeddings file already exists, it will be loaded.
    Otherwise, embeddings will be generated using the specified embedding model and saved to the file.

    Args:
        documents (list): A list of document fragments.

    Returns:
        numpy.ndarray: An array of embeddings corresponding to the input documents.
    """
    if os.path.exists(EMBEDDINGS_FILE):
        print("Loading embeddings...")
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = np.load(f)
    else:
        print("Generating embeddings... please be patient, as it can take a long time...")
        '''
        As a point of reference: generating embeddings for 2154 document fragments took about
        3 hours on a Raspberry Pi 5 with 8GB of RAM and an SSD, when using mxbai-embed-large
        '''
        embeddings = []
        start_time = time.time()
        for content in documents:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=content)
            embeddings.append(response["embedding"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 60:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            if minutes > 60:
                hours = int(minutes // 60)
                minutes = int(minutes % 60)
                print(f"Embeddings generated in {hours}h {minutes}m {seconds}s.")
            else:
                print(f"Embeddings generated in {minutes}m {seconds}s.")
        else:
            print(f"Embeddings generated in {elapsed_time}s.")
        print("Saving embeddings...")
        with open(EMBEDDINGS_FILE, 'wb') as f:
            np.save(f, embeddings)
    return embeddings

# Function to get relevant context from the vault based on user input
def get_relevant_context(input, embeddings, content, top_k=3):
    if embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [content[idx].strip() for idx in top_indices]
    return relevant_context

def ollama_chat(user_input, system_message, embeddings, content, ollama_model, conversation_history):
    """
    Conducts a chat conversation with the Ollama model.

    Args:
        user_input (str): The user's input message.
        system_message (str): The system's message.
        embeddings: The embeddings used for retrieving relevant context.
        content: The content used for retrieving relevant context.
        ollama_model: The Ollama model used for generating responses.
        conversation_history (list): The list of previous conversation messages.

    Returns:
        str: The assistant's response.

    """
    conversation_history.append({"role": "user", "content": user_input})
    
    # if len(conversation_history) > 1:
    #     query_json = {
    #         "Query": user_input,
    #         "Rewritten Query": ""
    #     }
    #     rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
    #     rewritten_query_data = json.loads(rewritten_query_json)
    #     rewritten_query = rewritten_query_data["Rewritten Query"]
    #     print(PINK + "Original Query: " + user_input + RESET_COLOR)
    #     print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    # else:
    rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, embeddings, content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Main script
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Local RAG Experiment")
    parser.add_argument("--regen", action='store_true', help="Whether to regenerate the embeddings or not")
    args = parser.parse_args()

    #if args.regen is present then delete DOCUMENTS_FILE and EMBEDDINGS_FILE
    if args.regen is True:
        if os.path.exists(DOCUMENTS_FILE):
            os.remove(DOCUMENTS_FILE)
        if os.path.exists(EMBEDDINGS_FILE):
            os.remove(EMBEDDINGS_FILE)

    print("Getting documents...")
    documents = load_documents()
    print("Getting embeddings...")
    embeddings = generate_embeddings(documents)

    # Convert to tensor and print embeddings
    print("Converting embeddings to tensor...")
    embeddings_tensor = torch.tensor(embeddings) 
    print("Embeddings for each document fragment:")
    print(embeddings_tensor)

    system_message = """
You are a helpful assistant for question-answering tasks for a single user. Use the supplied context to answer the question. All the context is relevant to the user's interests and work. Use the context to answer the question as best as you can.
Bring in extra relevant information you know to the user query from outside the given context. If you don't know the answer, just say that you don't know.
If you do know the answer, keep the answer concise. Bullet points and numbered lists are encouraged, where they are appropriate. 
"""

    conversation_history = []

    try:
        while True:
            query = input("\n\n\033[94mAsk Dingus² 🤖 ::::: \033[00m")
            # if query is bye then exit
            if query.lower() == 'bye':
                print("Exiting...")
                break
            
            # Retrieve and print the answer
            answer = ollama_chat(query, system_message, embeddings_tensor, documents, OLLAMA_MODEL, conversation_history)
            if answer is not None:
                print(answer)

    except KeyboardInterrupt:
        print("Script interrupted. Exiting gracefully.")
