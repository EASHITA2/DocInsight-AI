from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("=" * 60)
print("STEP 1: Loading PDF")
print("=" * 60)

# Load the PDF
loader = PyPDFLoader("data/raw_pdfs/CAT_Formulas_PDF.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} pages")

print("\nSTEP 2: Splitting into Chunks")

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("\nSTEP 3: Loading Embedding Model")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded Successfully!")

print("\nSTEP 4: Creating FAISS Vector Store")

# Create FAISS Vector Database
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

print("FAISS Vector Store Created Successfully!")

print("\nSTEP 5: Searching")

query = "Profit Formula"

results = vectorstore.similarity_search(
    query=query,
    k=3
)

print("\nTop 3 Results\n")

for i, doc in enumerate(results, start=1):
    print("=" * 60)
    print(f"Result {i}")
    print("=" * 60)
    print(doc.page_content[:500])
    print("\nMetadata:")
    print(doc.metadata)
    print()