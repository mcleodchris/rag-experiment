import os
import glob
from bs4 import BeautifulSoup
from docx import Document
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import ollama
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
import pickle

# Load environment variables from .env file
load_dotenv()

# Define constants
INDEX_FILE = os.environ.get('INDEX_FILE',"local_files.index")
DOCUMENTS_FILE = os.environ.get('DOCUMENTS_FILE',"documents.pkl")
EMBEDDINGS_FILE = os.environ.get('EMBEDDINGS_FILE',"embeddings.npy")
TOKEN_MODEL = os.environ.get('TOKEN_MODEL',"sentence-transformers/paraphrase-MiniLM-L6-v2")
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL','phi3:mini')

# Load functions
def load_documents():
    with open(DOCUMENTS_FILE, 'rb') as f:
        return pickle.load(f)

def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKEN_MODEL)

def load_model():
    return AutoModel.from_pretrained(TOKEN_MODEL)

def load_index():
    # Load the index from disk
    index = faiss.read_index(INDEX_FILE)

    # Load the documents, tokenizer, and model from disk
    documents = load_documents()
    tokenizer = load_tokenizer()
    model = load_model()

    return documents, tokenizer, model, index

# File extraction functions
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
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

# Load and embed documents
def embed(documents, tokenizer, model):
    # Join tokens into a single string for each document
    documents = [" ".join(doc) for doc in documents]
    # Tokenize all the documents at once
    encodings = tokenizer.batch_encode_plus(
        documents,
        truncation=True,
        padding=True,
        max_length=int(os.environ.get('ENCODING_MAX_LENGTH', 512)),
        return_tensors='pt'
    )

    # Pass the encodings to the model to generate embeddings
    with torch.no_grad():
        model_output = model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask']
        )

    # Use the mean of the last hidden state as the document embedding
    embeddings = model_output.last_hidden_state.mean(dim=1).numpy()

    return embeddings

def retrieve(query, index, documents, tokenizer, model, k):
    text=[query]
    print(f"Retrieving {k} most relevant documents")
    query_embedding = embed(text, tokenizer, model)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def generate_answer(query, index, documents, tokenizer, model):
    system_prompt = """
You are a helpful assistant for question-answering tasks for a single user. Use the supplied context to answer the question. All the context is relevant to the user's interests and work. Use the context to answer the question as best as you can.
Bring in extra relevant information you know to the user query from outside the given context. If you don't know the answer, just say that you don't know.
If you do know the answer, keep the answer concise. Bullet points and numbered lists are encouraged, where they are appropriate. 
"""

    relevant_docs = retrieve(query, index, documents, tokenizer, model, int(os.environ.get('RETRIEVAL_K', 15)))
    print(f"Found {len(relevant_docs)} relevant documents.")
    context = " ".join([" ".join(doc) for doc in relevant_docs])
    input_text = f"Context: {context}\n\nQuestion: {query.replace('show debugging info', '')}\n\nAnswer:"
    print("Sending everything to ollama...")
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': input_text}]
    )
    total_duration_ns = response['total_duration']

    # Convert the duration from nanoseconds to seconds
    total_duration_s = total_duration_ns / 1_000_000_000

    # Use the divmod function to convert the total seconds into minutes and seconds
    minutes, seconds = divmod(total_duration_s, 60)
    
    if "show debugging info" in query.lower():
        print("\n-------------------------\n")
        print("\033[91m" + input_text + "\033[00m")
        print("\n-------------------------\n")
        print(response)
        print("\n-------------------------\n")
        print("\n\n\033[94mYou Asked Dingus 🤖 ::::: \033[00m" + query)
    print(f"\n\n\033[94mResponse took: {int(minutes)} minutes and {seconds:.2f} seconds\033[00m")
    print("\n\n\033[94m" + response['message']['content'] + "\033[00m")
    

def generate_index():
    print("Generating index... please be patient...")
    # Retrieve file paths from environment variable
    file_paths = glob.glob(os.environ.get('DOCS_LOCATION', ''), recursive=True)
    print(f"Found {len(file_paths)} files.")
    documents = []
    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text:
                documents.append(word_tokenize(text))  # Use NLTK's word_tokenize
        except ValueError as e:
            print(e)
    print(f"Processed compatible {len(documents)} documents.")
    #pickle documents
    with open('documents.pkl', 'wb') as f:
        print("Pickling documents...")
        pickle.dump(documents, f)

    tokenizer = load_tokenizer()
    model = load_model()

    # Generate embeddings for all documents in batches of 50
    batch_size = os.environ.get('BATCH_SIZE', 50)
    num_documents = len(documents)
    embeddings = []
    for i in range(0, num_documents, batch_size):
        end_index = min(i+batch_size, num_documents)
        print(f"Embedding documents {i} to {end_index} of {num_documents}")
        batch_documents = documents[i:i+batch_size]
        batch_embeddings = embed(batch_documents, tokenizer, model)
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)

    # cache embeddings to disk
    print("Caching embeddings to disk...")
    np.save(EMBEDDINGS_FILE, embeddings)

    # Initialize a new FAISS index
    d = embeddings.shape[1]  # dimension
    nlist = 50  # number of Voronoi cells (clusters)
    k = 100  # number of nearest neighbors to use for training
    print("Initializing FAISS quantizer")
    quantizer = faiss.IndexFlatL2(d)  # the quantizer defines the Voronoi cells
    print("Initializing FAISS index")
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    
    # Train the index
    assert not index.is_trained
    print("Training FAISS index")
    index.train(embeddings)
    assert index.is_trained

    # Add the embeddings to the index
    print("Adding embeddings to the index")
    index.add(embeddings)

    # save the index to disk
    print("Saving index to disk")
    faiss.write_index(index, "local_files.index")
    return documents, tokenizer, model, index

# Main script
if __name__ == "__main__":
    # if local_files.index exists, load it, otherwise run generate_index() and then load it
    if os.path.exists(INDEX_FILE):
        documents, tokenizer, model, index = load_index()
    else:
        documents, tokenizer, model, index = generate_index()

    try:
        while True:
            query = input("\n\n\033[94mAsk Dingus 🤖 ::::: \033[00m")
            # if query is bye then exit
            if query.lower() == 'bye':
                print("Exiting...")
                break
            # if query is regenerate index then regenerate index and load it
            if query.lower() == 'regenerate index':
                print("Regenerating index... please be VERY patient...")
                documents, tokenizer, model, index = generate_index()
                print("Index regenerated.")
                query = input("\n\n\033[94mAsk Dingus 🤖 ::::: \033[00m")            
            
            # Retrieve and print the answer
            answer = generate_answer(query, index, documents, tokenizer, model)
            if answer is not None:
                print(answer)

    except KeyboardInterrupt:
        print("Script interrupted. Exiting gracefully.")
