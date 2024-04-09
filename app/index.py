import logging
import json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "aari1995/German_Semantic_STS_V2"
FORUM_POSTS = "../../scraper/forum_posts.json"
TEXT_FILE = "../res/forum_posts.txt"


def convert_html_to_text(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    return soup.get_text().strip().replace("\r", "").replace("\n\n", "\n")


def preprocess_html(data):
    for item in data:
        item["title"] = convert_html_to_text(item["title"])
        for post in item["posts"]:
            post["post"] = convert_html_to_text(post["post"])
    return data


def format_data(data):
    output_text = ""
    for item in data:
        if output_text != "":
            output_text += "\n\n"
        output_text += "Thema:" + item["title"] + "\n"
        for post in item["posts"]:
            poster = post["poster"] if post["poster"] is not None else "Unknown"
            output_text += poster + ": " + post["post"] + "\n\n"
    return output_text


def save_file(text):
    f = open(TEXT_FILE, "w")
    f.write(text)
    f.close()


def token_length_function(text_input, tokenizer):
    return len(tokenizer.encode(text_input, add_special_tokens=False))


def split_text(text, embedding_model):
    chunk_size = embedding_model.get_max_seq_length()
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=(lambda text: token_length_function(text, tokenizer)),
        separators=["\n\n\n", "\n\n", "\n"],
    )
    return splitter.split_text(text)


def embed_chunks(chunks, embedding_model):
    return embedding_model.encode(
        chunks, normalize_embeddings=True, show_progress_bar=True
    )


def index_data(chunks, embeddings):
    # Initialize an instance of HuggingFaceEmbeddings with the custom embedding model
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model_wrapper = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    text_embedding_pairs = zip(chunks, embeddings)
    return FAISS.from_embeddings(text_embedding_pairs, embedding_model_wrapper)


def main():
    logger.info(f"Opening input file: {FORUM_POSTS} ...")
    with open(FORUM_POSTS, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Reduce the size of the dataframe according to size limitations
    data = data[0:100]
    # Format the html inside the dictionary and remove trailing whitespaces
    logger.info(f"Preprocessing data: {len(data)} items ...")
    data = preprocess_html(data)
    # Rewrite the forum topics as a cursive conversation
    logger.info(f"Formatting data ...")
    formatted_text = format_data(data)
    # Save the cursive text file
    logger.info(
        f"Saving formatted text of size {len(formatted_text)} under {TEXT_FILE} ..."
    )
    save_file(formatted_text)
    # Retrieve the embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)
    # Split the formatted text in chunks dependent on model input window token size
    logger.info(f"Splitting text in chunks ...")
    chunks = split_text(formatted_text, model)
    # Embed the chunks
    logger.info(f"Embedding {len(chunks)} chunks ...")
    embedding_output = embed_chunks(chunks, model)
    # Index the embeddings corresponding to the chunks
    logger.info(f"Indexing the embeddings ...")
    vector_store = index_data(chunks, embedding_output)
    # Save the vector store locally
    logger.info(
        f"Saving the vector store of size {len(vector_store.index_to_docstore_id)} locally ..."
    )
    vector_store.save_local("../res/forum_index")


if __name__ == "__main__":
    main()
