from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retrieval_pipeline import load_vector_store

load_dotenv()


def get_llm():
    print("Loading LLM...")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    return llm


def rewrite_query(llm, chat_history, query):

    if len(chat_history) == 0:
        return query

    history = "\n".join(chat_history[-6:])

    rewrite_prompt = f"""
You are a query rewriting assistant.

Given the conversation history and latest user question,
rewrite the latest question into a complete standalone question.

Only return the rewritten question.
Do not explain anything.

Chat History:
{history}

Latest Question:
{query}

Standalone Question:
"""

    rewritten_query = llm.invoke(rewrite_prompt).content.strip()

    return rewritten_query


def retrieve_documents(vector_store, llm):

    print("\nHistory Aware Conversational RAG Chatbot")
    print("Type 'exit' to quit\n")

    chat_history = []

    while True:

        query = input("Enter Query: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Rewrite query using history
        standalone_query = rewrite_query(
            llm,
            chat_history,
            query
        )

        print("\nRewritten Query:")
        print(standalone_query)

        # Retrieve documents
        results = vector_store.similarity_search(
            query=standalone_query,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        history = "\n".join(chat_history[-6:])

        # Generate answer
        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not available in the context,
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

        # Debug retrieved chunks
        print("\nRetrieved Context:")

        for i, doc in enumerate(results):
            print(f"\nChunk {i + 1}:")
            print(doc.page_content)

        print("\n" + "-" * 60)


def main():

    llm = get_llm()

    vector_store = load_vector_store()

    retrieve_documents(
        vector_store,
        llm
    )


if __name__ == "__main__":
    main()