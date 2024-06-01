import argparse
import glob
import json
import os
import re
import shutil
import time


from dotenv import load_dotenv
from openai import OpenAI
import ollama

from util import (
    format_elapsed_time,
    extract_text,
    AnsiColours as ansi,
    get_database,
    load_or_generate,
    rewrite_query,
)

"""
 This version of Dingus (v2) is a smooshing together of ideas from Dingus v1,
 code by All About AI, and an example from Ollama:
 - All About AI Video: https://www.youtube.com/watch?v=Oe-7dGDyzPM
 - All About AI Repo: https://github.com/AllAboutAI-YT/easy-local-rag
 - Ollama Example: https://ollama.com/blog/embedding-models

"""


def chunk_text(text):
    # remove any YAML front-matter from the text
    text = re.sub(r"---\n.*\n---\n", "", text)
    # split the text into paragraphs
    paragraphs = re.split(r"\n\n+", text)
    return paragraphs


def generate_documents():
    """
    Generate a list of pre-processed documents from the specified file paths.

    Returns:
        list: A list of pre-processed document fragments.

    Raises:
        ValueError: If an error occurs during text extraction.
    """
    print("No pre-processed documents found. Loading from disk...")
    documents = []
    file_paths = []
    if "," in LOCAL_DOCUMENTS:
        segments = LOCAL_DOCUMENTS.split(",")
        for segment in segments:
            file_paths.extend(glob.glob(segment.strip(), recursive=True))
    else:
        file_paths = glob.glob(LOCAL_DOCUMENTS, recursive=True)
    for file_path in file_paths:
        try:
            text = extract_text(file_path)
            if text:
                documents.extend(chunk_text(text))
        except ValueError as e:
            print(e)
    print(f"Processed {len(documents)} document fragments.")
    # remove any empty documents
    documents = [doc for doc in documents if doc]
    return documents


def generate_embeddings_chromadb(documents, collection, embedding_model):
    """
    Generate embeddings for a list of documents and store them in a vector embedding database.

    Args:
        documents (list): A list of documents to generate embeddings for.
        collection: The vector embedding database collection to store the embeddings in.

    Returns:
        None
    """
    if collection.count() == 0:
        print(
            "Generating embeddings... please be patient, as it can take a long time..."
        )
        start_time = time.time()
        for i, d in enumerate(documents):
            response = ollama.embeddings(model=embedding_model, prompt=d)
            embedding = response["embedding"]
            collection.add(ids=[str(i)], embeddings=[embedding], documents=[d])
        end_time = time.time()
        elapsed_time = format_elapsed_time((end_time - start_time))
        print(f"Embeddings generated in {elapsed_time}.")
    else:
        print("Embeddings already exist in the database.")


def get_relevant_context_chromadb(query, collection, embedding_model, top_k=3):
    """
    Retrieves the most relevant document from a collection based on a query.

    Args:
        query (str): The query string used to generate an embedding for the prompt.
        collection (Collection): The collection object used to query the database.

    Returns:
        list: A list of the most relevant documents retrieved from the collection.
    """
    # generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(prompt=query, model=embedding_model)
    results = collection.query(
        query_embeddings=[response["embedding"]], n_results=top_k
    )["documents"][0]
    # dump the results structure to the console

    return results


def generate_answer(
    user_input,
    conversation_history,
    system_message,
    collection,
    embedding_model,
    ollama_model,
    relevant_results=3,
):
    """
    Generates an answer based on the user input, conversation history, system message, and collection.

    Args:
        user_input (str): The user's input.
        conversation_history (list): A list of dictionaries representing the conversation history.
        system_message (str): The system message.
        collection (str): The collection.

    Returns:
        str: The generated answer.
    """
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {"Query": user_input, "Rewritten Query": ""}
        rewritten_query_json = rewrite_query(
            json.dumps(query_json), conversation_history, api, OLLAMA_MODEL
        )
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(ansi.PINK + "Original Query: " + user_input + ansi.RESET_COLOR)
        print(ansi.PINK + "Rewritten Query: " + rewritten_query + ansi.RESET_COLOR)
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context_chromadb(
        rewritten_query, collection, embedding_model, relevant_results
    )
    context_str = ""
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print(
            "Context Pulled from Documents: \n\n"
            + ansi.CYAN
            + context_str
            + ansi.RESET_COLOR
        )
    else:
        print(ansi.CYAN + "No relevant context found." + ansi.RESET_COLOR)

    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str

    conversation_history[-1]["content"] = user_input_with_context

    messages = [{"role": "system", "content": system_message}, *conversation_history]

    response = api.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )

    conversation_history.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    # total_duration_ns = response.choices[0].total_duration
    # # print the total duration formatted with format_elapsed_time
    # print(
    #     ansi.GREEN
    #     + "Total duration: "
    #     + format_elapsed_time(total_duration_ns / 1_000_000_000)
    #     + ansi.RESET_COLOR
    # )

    return response.choices[0].message.content


# Main script
if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv(override=True)

    DOCUMENTS_FILE = os.environ.get("DOCUMENTS_FILE", "documents2.pkl")
    EMBEDDINGS_FILE = os.environ.get("EMBEDDINGS_FILE", "embeddings2.npy")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
    LOCAL_DOCUMENTS = os.environ.get("DOCS_LOCATION", "./docs/**/*.txt")
    OPENAI_API_BASE = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    CHROMADB_FILE = os.environ.get("CHROMADB_FILE", "chromadb.db")
    RELEVANT_RESULTS = int(os.environ.get("RELEVANT_RESULTS", 3))

    print(" ::: Environment :::")
    print(f"DOCUMENTS_FILE: {DOCUMENTS_FILE}")
    print(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"LOCAL_DOCUMENTS: {LOCAL_DOCUMENTS}")
    print(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
    print(f"CHROMADB_FILE: {CHROMADB_FILE}")
    print(f"RELEVANT_RESULTS: {RELEVANT_RESULTS}")

    parser = argparse.ArgumentParser(description="Local RAG Experiment")
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Whether to regenerate the embeddings or not",
    )
    args = parser.parse_args()

    # if args.regen is present then delete DOCUMENTS_FILE and EMBEDDINGS_FILE
    if args.regen is True:
        # remove CHROMADB_FILE then recreate
        if os.path.exists(CHROMADB_FILE):
            shutil.rmtree(CHROMADB_FILE)
        # remove DOCUMENTS_FILE
        if os.path.exists(DOCUMENTS_FILE):
            os.remove(DOCUMENTS_FILE)

    api = OpenAI(base_url=OPENAI_API_BASE, api_key=OLLAMA_MODEL)

    db, collection = get_database(CHROMADB_FILE)

    print("Getting documents...")
    documents = load_or_generate(DOCUMENTS_FILE, generate_documents, [])
    print("Getting embeddings...")
    # embeddings = generate_embeddings(documents)

    # # Convert to tensor and print embeddings
    # print("Converting embeddings to tensor...")
    # embeddings_tensor = torch.tensor(embeddings)
    # print("Embeddings for each document fragment:")
    # print(embeddings_tensor)

    generate_embeddings_chromadb(documents, collection, EMBEDDING_MODEL)

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
            if query.lower() == "bye":
                print("Exiting...")
                break

            # Retrieve and print the answer
            # answer = ollama_chat(query, system_message, embeddings_tensor, documents, OLLAMA_MODEL, conversation_history)
            answer = generate_answer(
                query,
                conversation_history,
                system_message,
                collection,
                EMBEDDING_MODEL,
                OLLAMA_MODEL,
                RELEVANT_RESULTS,
            )
            if answer is not None:
                print(answer)

    except KeyboardInterrupt:
        print("Script interrupted. Exiting gracefully.")
