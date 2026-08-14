import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# ============================================================
# Load API Key
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

print("Gemini Loaded Successfully!")

# ============================================================
# Load Embedding Model
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded!")

# ============================================================
# Load FAISS Vector Database
# ============================================================

vectorstore = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("Vector Database Loaded!")

# ============================================================
# Chat Loop
# ============================================================

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    docs = vectorstore.similarity_search(
        question,
        k=3
    )

    context = ""

    print("\n")
    print("=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, doc in enumerate(docs, start=1):

        page = doc.metadata.get("page", "Unknown")

        print(f"Source {i} → Page {page + 1}")

        context += doc.page_content
        context += "\n\n"

    print("\n")
    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    print(context)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided in the context.

If the answer is not present, reply exactly:

I couldn't find that information in the document.

Context:

{context}

Question:

{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    print("\n")
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response.text)