from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("=" * 60)
print("BUILDING VECTOR DATABASE")
print("=" * 60)

# Load PDF
loader = PyPDFLoader("data/raw_pdfs/CAT_Formulas_PDF.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Loaded {len(chunks)} chunks")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("Vector Store Created")

# Save to disk
vectorstore.save_local("vector_store")

print("\nVector Store Saved Successfully!")