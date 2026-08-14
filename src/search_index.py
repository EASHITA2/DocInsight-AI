from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("=" * 60)
print("LOADING SAVED VECTOR DATABASE")
print("=" * 60)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load saved FAISS index
vectorstore = FAISS.load_local(
    folder_path="vector_store",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

print("Vector Store Loaded Successfully!")

while True:
    query = input("\nEnter your question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    results = vectorstore.similarity_search(
        query=query,
        k=3
    )

    print("\nTop 3 Results:\n")

    for i, doc in enumerate(results, start=1):
        print("=" * 60)
        print(f"Result {i}")
        print("=" * 60)
        print(doc.page_content)
        print("\nMetadata:")
        print(doc.metadata)
        print()