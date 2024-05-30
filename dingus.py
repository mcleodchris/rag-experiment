import os
import glob
from bs4 import BeautifulSoup
from docx import Document
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import ollama
from dotenv import load_dotenv

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
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".html":
        return extract_text_from_html(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)

# Load and embed documents
def embed(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def retrieve(query, index, documents, tokenizer, model, k=5):
    text=query
    query_embedding = embed(text, tokenizer, model).numpy()
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def generate_answer(query, index, documents, tokenizer, model):
    system_prompt = """
You are an assistant for question-answering tasks for a single user. Use the supplied context to answer the question. All the context is relevant to the user's interests and work. Use the context to answer the question as best as you can.
If you don't know the answer, just say that you don't know.
If you do know the answer, keep the answer concise. Bullet points and numbered lists are encouraged, where they are appropriate. 
"""

    relevant_docs = retrieve(query, index, documents, tokenizer, model)
    context = " ".join(relevant_docs)
    input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.chat(
        model='phi3:mini',
        messages=[{'role': 'system', 'content': system_prompt},{'role': 'user', 'content': input_text}]
    )
    print("\n-------------------------\n")
    print("\033[91m" + input_text + "\033[00m") # for debugging
    print("\n-------------------------\n")
    print(response) # for debugging
    print("\n-------------------------\n")
    print("\n\n\033[94m" + response['message']['content'] + "\033[00m")
    


def answer_query(query, index, documents, tokenizer, model):
    return generate_answer(query, index, documents, tokenizer, model)

# Main script
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve file paths from environment variable
    file_paths = glob.glob(os.environ.get('DOCS_LOCATION', ''), recursive=True)

    documents = []
    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text:
                documents.append(text)
        except ValueError as e:
            print(e)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

    embeddings = []
    for doc in documents:
        embeddings.append(embed(doc, tokenizer, model).numpy())
    embeddings = np.vstack(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    try:
        while True:
            query = input("\n\n\033[94mAsk Dingus 🤖 ::::: \033[00m")
            print(answer_query(query, index, documents, tokenizer, model))
    except KeyboardInterrupt:
        print("Script interrupted. Exiting gracefully.")
