from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()


def load_vector_store(persist_directory="db/Semantic_db"):
    print("Loading vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store







def main():

    vector_store = load_vector_store()


if __name__ == "__main__":
    main()