from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()


def load_vector_store(persist_directory="db/Chroma_db"):
    print("Loading vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vector_store


def get_llm():
    print("Loading LLM...")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm


def retrieve_documents(vector_store, llm):

    print("\nConversational RAG Chatbot")
    print("Type 'exit' to quit\n")

    chat_history = []

    while True:

        query = input("Enter Query: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Last few messages only
        history = "\n".join(chat_history[-6:])

        results = vector_store.similarity_search(
            query=query,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        prompt = f"""
You are a helpful AI assistant.

Use chat history and context to answer.

If answer is not present in context,
say: "I don't have enough information."

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        print("\nAnswer:")
        print(response.content)

        # Store conversation
        chat_history.append(f"User: {query}")
        chat_history.append(f"Assistant: {response.content}")

        print("\nRetrieved Context:")

        for i, doc in enumerate(results):
            print(f"\nChunk {i+1}:")
            print(doc.page_content)

        print("\n" + "-" * 60)


def main():

    vector_store = load_vector_store()

    llm = get_llm()

    retrieve_documents(
        vector_store,
        llm
    )


if __name__ == "__main__":
    main()