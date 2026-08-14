from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("data/raw_pdfs/CAT_Formulas_PDF.pdf")
documents = loader.load()

# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Create embeddings for ALL chunks
# -----------------------------
print("\nGenerating embeddings for all chunks...")

chunk_embeddings = model.encode(
    [chunk.page_content for chunk in chunks]
)

print("Done!")

# -----------------------------
# User Query
# -----------------------------
query = "Profit formula"

query_embedding = model.encode([query])

# -----------------------------
# Compute cosine similarity
# -----------------------------
similarities = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]

# -----------------------------
# Find best match
# -----------------------------
best_index = similarities.argmax()

print("\nMost Relevant Chunk\n")

print(chunks[best_index].page_content)

print("\nSimilarity Score:")

print(similarities[best_index])