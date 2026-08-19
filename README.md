# 📄 DocInsight AI

### Turn documents into answers, insights, and interactive learning with RAG.

DocInsight AI is an end-to-end **Retrieval-Augmented Generation (RAG)** application that enables users to upload one or multiple PDF documents and interact with their content using natural language.

The project was built to explore the complete RAG pipeline — from document ingestion and chunking to embeddings, vector retrieval, context augmentation, grounded generation, source attribution, and AI-powered practice generation.

---

## 🚀 Features

### 💬 Document-Grounded Q&A
Ask natural-language questions about uploaded PDFs and receive answers grounded in retrieved document context.

### 📚 Multi-PDF Support
Upload and query multiple PDF documents within the same workspace.

### 🔎 Semantic Search
Uses vector embeddings and FAISS similarity search to retrieve semantically relevant document chunks rather than relying only on keyword matching.

### 📖 Source Attribution
Answers can be traced back to:
- Source PDF
- Page number
- Retrieved document chunk
- Page preview

### 📝 AI Document Summarization
Generate structured summaries highlighting:
- Main topics
- Key concepts
- Important findings
- Conclusions

### 💡 Suggested Questions
Automatically generates document-grounded questions to help users explore uploaded material.

### 🧮 Numerical & Application Reasoning
The system can retrieve formulas, methods, and concepts from documents and apply them to new numerical problems or scenarios.

### 🧠 Case-Study Reasoning
Document-supported concepts can be applied to new case studies and reasoning-based questions.

### 🎯 Interactive Practice Generator
Generate customized practice sets containing:
- MCQs
- Numerical questions
- Conceptual questions
- Case studies

Users can select question type, difficulty, and number of questions.

### ✅ AI-Assisted Evaluation
Practice responses can be evaluated with:
- Correct/incorrect feedback
- Expected answers
- Explanations
- Quiz scores
- Accuracy
- Source references

---

# 🧠 RAG Architecture

The core pipeline follows:

```text
PDF Documents
      ↓
PDF Parsing
      ↓
Recursive Text Chunking
      ↓
HuggingFace Embeddings
      ↓
FAISS Vector Index
      ↓
User Query
      ↓
Semantic Similarity Search
      ↓
Relevant Document Chunks
      ↓
Context Augmentation
      ↓
Google Gemini
      ↓
Grounded Response
      ↓
Source / Page Attribution
```

---

## 🔬 How the RAG Pipeline Works

### 1. Document Ingestion

Uploaded PDFs are parsed using LangChain's `PyPDFLoader`.

Metadata such as the original PDF filename and page information is preserved so retrieved information can later be traced back to its source.

### 2. Recursive Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The overlap helps preserve contextual continuity when information spans chunk boundaries.

### 3. Embedding Generation

Document chunks are converted into dense vector representations using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model maps semantically related text into nearby regions of vector space.

### 4. FAISS Vector Retrieval

The generated embeddings are indexed using **FAISS**.

When a user submits a question, semantic similarity search retrieves the most relevant document chunks.

### 5. Context Augmentation

Retrieved chunks, source information, page metadata, conversation context, and the current question are assembled into the generation prompt.

### 6. Grounded Generation

**Google Gemini** generates the final response using the retrieved document context as its primary knowledge source.

The prompting strategy distinguishes between:

- Direct document questions
- Numerical/application problems
- Case-study reasoning
- Insufficient document knowledge

### 7. Source Traceability

Retrieved metadata is preserved throughout the pipeline, allowing DocInsight AI to display the source PDF and page associated with retrieved evidence.

---

# 🏗️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| LLM | Google Gemini |
| RAG Framework | LangChain |
| Embeddings | HuggingFace Sentence Transformers |
| Embedding Model | `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| PDF Loading | PyPDFLoader |
| PDF Rendering | PyMuPDF |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```text
DocInsight-AI/
│
├── app.py
├── src/
│   ├── llm.py
│   └── rag.py
│
├── assets/
│   └── docinsight_logo.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact structure may evolve as the project is further modularized.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/EASHITA2/DocInsight-AI.git
cd DocInsight-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

The `.env` file should **never be committed to GitHub**.

It is excluded through `.gitignore`.

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Then open the Streamlit application in your browser.

---

# 🔄 Example Workflow

```text
1. Upload one or more PDFs
             ↓
2. Documents are parsed and chunked
             ↓
3. Chunks are embedded and indexed in FAISS
             ↓
4. Ask a question
             ↓
5. Relevant chunks are retrieved
             ↓
6. Gemini receives the retrieved context
             ↓
7. A grounded answer is generated
             ↓
8. Source PDF and page information are displayed
```

---

# 💡 What I Learned

This project was built as a hands-on exploration of **Retrieval-Augmented Generation**.

Some of the key areas explored while building DocInsight AI include:

- Document ingestion
- Chunking strategies
- Chunk overlap and context preservation
- Sentence-transformer embeddings
- Semantic similarity
- Vector indexing and retrieval
- Context augmentation
- Prompt engineering
- Grounded LLM generation
- Metadata preservation
- Source attribution
- Multi-document retrieval
- Conversational RAG
- LLM-based evaluation

One of the main lessons from the project is that RAG performance depends on much more than the LLM itself.

The quality of:

**Parsing → Chunking → Embeddings → Retrieval → Context Construction → Generation**

can directly affect the quality and grounding of the final response.

---

# 🔭 Future Improvements

Some areas I plan to explore further:

- Hybrid semantic + keyword retrieval
- Cross-encoder reranking
- Adaptive/document-aware chunking
- Retrieval quality evaluation
- RAG evaluation metrics
- Improved confidence scoring
- Persistent vector indexes
- Larger document collections
- Improved citation verification

---

# 👩‍💻 Project

**DocInsight AI**

Built as a hands-on project to explore the architecture and engineering behind modern Retrieval-Augmented Generation systems.

⭐ If you find the project useful, feel free to star the repository.