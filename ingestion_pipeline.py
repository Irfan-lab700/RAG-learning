import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

    
def load_documents(docs_path = "docs"):
    print("Loading documents...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if(len(documents) == 0):
        raise ValueError(f"No documents found in directory '{docs_path}'.")
    for i,doc in enumerate(documents[:3]):
        print(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:500]}...")
        print(f"Content_length: {len(doc.page_content)} characters")
    return documents


def split_documents(documents):
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator = "\n"
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}")
        print(chunk.page_content[:500])
        print(f"Length: {len(chunk.page_content)}")

    return chunks


def create_vector_store(chunks,persist_directory="db/Chroma_db"):
    print("Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata = {"hnsw:space":"cosine"}
    )
    return vector_store

def main():
    print("Hye Irfan")
    documents = load_documents(docs_path = "docs")
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)

if __name__ == "__main__":
    main()