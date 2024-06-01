import json
import pickle
from bs4 import BeautifulSoup
import chromadb
from docx import Document
import os


# Constants
# ANSI escape codes for colours
class AnsiColours:
    NEON_GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PINK = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    WHITE = "\033[97m"
    RESET_COLOR = "\033[0m"


def format_elapsed_time(elapsed_time):
    """
    Formats the elapsed time into a human-readable string.

    Args:
        elapsed_time (float): The elapsed time in seconds.

    Returns:
        str: The formatted elapsed time string.

    """
    if elapsed_time > 60:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        if minutes > 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
            output = f"{hours}h {minutes}m {seconds}s"
        else:
            output = f"{minutes}m {seconds}s"
    else:
        output = f"{elapsed_time}s"
    return output


# File extraction functions
def extract_text_from_html(file_path):
    """
    Extracts text content from an HTML file.

    Args:
        file_path (str): The path to the HTML file.

    Returns:
        str: The extracted text content from the HTML file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
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
    with open(file_path, "r", encoding="utf-8") as file:
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
        ".md": extract_text_from_txt,
    }
    ext = os.path.splitext(file_path)[1].lower()
    extractor = extractors.get(ext)
    if extractor:
        return extractor(file_path)


def load_or_generate(filename, generate_function, default=None):
    """
    Load a file if it exists, otherwise generate it.

    Args:
        filename (str): The name of the file.
        generate_function (function): The function to generate the file.

    Returns:
        object: The loaded or generated object.

    """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        obj = generate_function()
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
        # Return the default value if the object is None
        if obj is None:
            return default
        return obj


def rewrite_query(user_input_json, conversation_history, client, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]]
    )
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})


def get_database(db_file):
    db = chromadb.PersistentClient(path=db_file)
    collection = db.get_or_create_collection(name="docs")
    return db, collection
