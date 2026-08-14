from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Step 1: Load the PDF
# -----------------------------
loader = PyPDFLoader("data/raw_pdfs/CAT_Formulas_PDF.pdf")
documents = loader.load()

print("=" * 60)
print(f"Original Documents (Pages): {len(documents)}")
print("=" * 60)

# -----------------------------
# Step 2: Create Text Splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# -----------------------------
# Step 3: Split into chunks
# -----------------------------
chunks = text_splitter.split_documents(documents)

print(f"\nTotal Chunks Created: {len(chunks)}")

# -----------------------------
# Step 4: Inspect the first chunk
# -----------------------------
print("\n" + "=" * 60)
print("FIRST CHUNK")
print("=" * 60)

print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)