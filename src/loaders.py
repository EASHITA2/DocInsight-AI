from langchain_community.document_loaders import PyPDFLoader

# Path to the PDF
pdf_path = "data/raw_pdfs/CAT_Formulas_PDF.pdf"

# Create loader
loader = PyPDFLoader(pdf_path)

# Load PDF
documents = loader.load()

print(f"Total pages loaded: {len(documents)}")

print("\nFirst Document:\n")
print(documents[0])