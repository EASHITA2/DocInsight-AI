from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("STEP 1: Loading PDF...")
print("=" * 60)

loader = PyPDFLoader("data/raw_pdfs/CAT_Formulas_PDF.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

print("\nSTEP 2: Splitting into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("\nSTEP 3: Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

print("\nSTEP 4: Generating embedding for the first chunk...")

embedding = model.encode(chunks[0].page_content)

print("\nEmbedding generated!")

print(f"\nEmbedding Dimension: {len(embedding)}")

print("\nFirst 20 values of the embedding:")

print(embedding[:20])

print("\nFirst Chunk Preview:\n")

print(chunks[0].page_content[:300])