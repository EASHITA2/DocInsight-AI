import os
import json
import tempfile
import base64
from pathlib import Path
import fitz
import streamlit as st
from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================================
# Page Config
# ==========================================================

ASSET_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSET_DIR / "docinsight_logo.png"

st.set_page_config(
    page_title="DocInsight AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ==========================================================
# Load Gemini
# ==========================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ==========================================================
# Load Embedding Model
# ==========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# Grouped Source Renderer
# ==========================================================

def render_grouped_sources(source_items, heading="📚 Sources"):
    if not source_items:
        return

    st.markdown(f"### {heading}")

    grouped = {}
    for item in source_items:
        filename = item.get("file", "PDF")
        grouped.setdefault(filename, []).append(item)

    for filename, items in grouped.items():
        pages = sorted({item.get("page", 1) for item in items})
        pages_text = ", ".join(str(p) for p in pages)
        page_word = "Page" if len(pages) == 1 else "Pages"

        with st.expander(
            f"📘 {filename} — {page_word} {pages_text}",
            expanded=False
        ):
            for index, item in enumerate(items, start=1):
                page = item.get("page", 1)
                confidence = item.get("confidence", 0)
                text = item.get("text", "")
                preview = item.get("preview") or text[:500]

                st.markdown(f"#### Page {page} • {confidence}% Match")
                st.markdown("**Preview**")
                st.write(preview + ("..." if len(text) > len(preview) else ""))

                with st.expander("View Full Retrieved Chunk"):
                    st.write(text)

                if "page_images" in st.session_state:
                    file_images = st.session_state.page_images.get(filename, [])
                    page_index = page - 1
                    if 0 <= page_index < len(file_images):
                        st.markdown("**📷 Page Preview**")
                        st.image(
                            file_images[page_index],
                            caption=f"{filename} — Page {page}",
                            width=500
                        )

                if index < len(items):
                    st.markdown("---")


# ==========================================================
# Session State
# ==========================================================

defaults = {
    "dark_mode": True,
    "messages": [],
    "suggested_questions": [],
    "selected_question": None,
    "generate_summary": False,
    "summary": "",
    "generate_practice": False,
    "practice_questions": "",
    "practice_answers": "",
    "show_practice_answers": False,
    "practice_data": [],
    "practice_index": 0,
    "practice_results": [],
    "practice_submitted": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:

    # ======================================================
    # Branding
    # ======================================================
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("## ✨ DocInsight AI")

    st.caption("Your AI document learning assistant")
    st.markdown("---")

    # ======================================================
    # Upload
    # ======================================================
    st.markdown("### 📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents."
    )

    if uploaded_files:

        st.caption(f"📚 {len(uploaded_files)} document(s) selected")

        if st.button(
            "📝 Summarize Documents",
            use_container_width=True
        ):
            st.session_state.generate_summary = True

        st.markdown("---")

        # ==================================================
        # Practice Generator
        # ==================================================
        st.markdown("### 🧠 Practice Generator")

        st.caption(
            "Create AI-generated questions from your uploaded documents."
        )

        practice_type = st.selectbox(
            "Question Type",
            ["Mixed", "MCQ", "Numerical", "Conceptual", "Case Study"],
            key="practice_type"
        )

        practice_difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard"],
            index=1,
            key="practice_difficulty"
        )

        practice_count = st.selectbox(
            "Number of Questions",
            [3, 5, 10],
            index=1,
            key="practice_count"
        )

        if st.button(
            "🎯 Generate Practice",
            use_container_width=True
        ):
            st.session_state.generate_practice = True

    st.markdown("---")

    # ======================================================
    # Chat Controls
    # ======================================================
    # ======================================================
# Appearance
# ======================================================
    st.markdown("### 🎨 Appearance")

    dark_mode = st.toggle(
    "🌙 Dark Mode",
    value=st.session_state.dark_mode
)

    st.session_state.dark_mode = dark_mode

    st.markdown("---")    
 # ==========================================================
# Main Page / Landing State
# ==========================================================

logo_uri = image_to_data_uri(LOGO_PATH)

# ==========================================================
# AI Product Landing State
# ==========================================================

if not uploaded_files:

    # ======================================================
    # Landing Page CSS
    # ======================================================

    st.markdown(
        """
<style>

/* ========================================================
   LANDING PAGE CONTAINER
======================================================== */

.di-hero {
    max-width: 1350px;
    width: 92%;
    margin: 35px auto 0 auto;
    padding: 35px 30px 60px 30px;
    text-align: center;
}


/* ========================================================
   LOGO
======================================================== */

.di-logo-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin-bottom: 4px;
}

.di-logo {
    width: 125px;
    height: 125px;
    object-fit: contain;
    display: block;
    margin: 0 auto 14px auto;

    animation:
        logoPop 0.8s cubic-bezier(.34,1.56,.64,1),
        logoFloat 4s ease-in-out 1s infinite;

    filter:
        drop-shadow(0px 12px 24px rgba(139,92,246,0.22));
}

.di-logo-fallback {
    font-size: 70px;
    text-align: center;
    line-height: 1;
    margin-bottom: 14px;

    animation:
        logoPop 0.8s cubic-bezier(.34,1.56,.64,1),
        logoFloat 4s ease-in-out 1s infinite;
}

@keyframes logoPop {
    0% {
        opacity: 0;
        transform: scale(0.35) translateY(25px);
    }

    70% {
        opacity: 1;
        transform: scale(1.10) translateY(-4px);
    }

    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

@keyframes logoFloat {
    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }
}


/* ========================================================
   TITLE
======================================================== */

.di-title {
    font-size: 64px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1.8px;

    margin-top: 4px;
    margin-bottom: 14px;

    background: linear-gradient(
        90deg,
        #8b5cf6,
        #6366f1,
        #3b82f6
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    animation: fadeUp 0.7s ease-out;
}


/* ========================================================
   SUBTITLE
======================================================== */

.di-subtitle {
    max-width: 900px;
    margin: 0 auto 24px auto;

    font-size: 20px;
    line-height: 1.65;

    color: #94a3b8;

    animation: fadeUp 0.9s ease-out;
}


/* ========================================================
   FEATURE CHIPS
======================================================== */

.di-chip-row {
    display: flex;
    justify-content: center;
    align-items: center;

    flex-wrap: wrap;
    gap: 10px;

    margin: 10px 0 38px 0;

    animation: fadeUp 1s ease-out;
}

.di-chip {
    padding: 8px 15px;

    border-radius: 999px;

    border: 1px solid rgba(139,92,246,0.25);

    background: rgba(139,92,246,0.08);

    font-size: 14px;
    font-weight: 600;

    color: #a78bfa;
}


/* ========================================================
   FEATURE GRID
======================================================== */

.di-features {
    display: grid;

    grid-template-columns: repeat(4, 1fr);

    gap: 22px;

    margin-top: 10px;

    text-align: left;

    animation: fadeUp 1.1s ease-out;
}


/* ========================================================
   FEATURE CARDS
======================================================== */

.di-card {
    padding: 28px;
    min-height: 190px;
    border-radius: 18px;

    border: 1px solid rgba(148,163,184,0.18);

    background: rgba(255,255,255,0.025);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        box-shadow 0.25s ease,
        background 0.25s ease;
}

.di-card:hover {
    transform: translateY(-6px);

    border-color: rgba(139,92,246,0.60);

    background: rgba(139,92,246,0.055);

    box-shadow:
        0 16px 40px
        rgba(0,0,0,0.13);
}

.di-card-icon {
    font-size: 27px;
    margin-bottom: 14px;
}

.di-card-title {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.di-card-text {
    color: #94a3b8;

    font-size: 14px;
    line-height: 1.55;
}


/* ========================================================
   CTA
======================================================== */

.di-cta {
    max-width: 650px;

    margin: 38px auto 0 auto;

    padding: 17px 24px;

    border-radius: 15px;

    border: 1px solid rgba(59,130,246,0.20);

    background: linear-gradient(
        90deg,
        rgba(139,92,246,0.08),
        rgba(59,130,246,0.08)
    );

    color: #94a3b8;

    font-size: 14px;
    line-height: 1.55;

    animation: fadeUp 1.25s ease-out;
}

.di-cta strong {
    font-size: 16px;
    color: #a78bfa;
}


/* ========================================================
   GENERAL ANIMATION
======================================================== */

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(18px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


/* ========================================================
   TABLET
======================================================== */

@media (max-width: 1000px) {

    .di-features {
        grid-template-columns: repeat(2, 1fr);
    }

    .di-title {
        font-size: 46px;
    }
}


/* ========================================================
   MOBILE
======================================================== */

@media (max-width: 650px) {

    .di-hero {
        padding: 20px 5px 35px 5px;
    }

    .di-title {
        font-size: 38px;
        letter-spacing: -1px;
    }

    .di-subtitle {
        font-size: 16px;
    }

    .di-features {
        grid-template-columns: 1fr;
    }

    .di-logo {
        width: 85px;
        height: 85px;
    }

    .di-logo-fallback {
        font-size: 60px;
    }
}

</style>
        """,
        unsafe_allow_html=True
    )

    # ======================================================
    # Logo
    # ======================================================

    if logo_uri:
        logo_html = (
            f'<div class="di-logo-wrap">'
            f'<img src="{logo_uri}" '
            f'class="di-logo" '
            f'alt="DocInsight AI Logo">'
            f'</div>'
        )
    else:
        logo_html = (
            '<div class="di-logo-wrap">'
            '<div class="di-logo-fallback">✨</div>'
            '</div>'
        )

    # ======================================================
    # Landing Page HTML
    #
    # IMPORTANT:
    # Keep this HTML left-aligned inside the string.
    # Indented HTML can be interpreted by Markdown as code.
    # ======================================================

    hero_html = f"""<div class="di-hero">
{logo_html}
<div class="di-title">DocInsight AI</div>
<div class="di-subtitle">Turn documents into knowledge with AI.<br>Upload one or more PDFs, ask grounded questions, generate summaries, solve new problems using document concepts, and practice with interactive quizzes.</div>
<div class="di-chip-row">
<div class="di-chip">💬 Ask</div>
<div class="di-chip">📝 Summarize</div>
<div class="di-chip">🧠 Practice</div>
<div class="di-chip">🔎 Search</div>
</div>
<div class="di-features">
<div class="di-card">
<div class="di-card-icon">💬</div>
<div class="di-card-title">Ask Anything</div>
<div class="di-card-text">Chat naturally with your PDFs and receive answers grounded in your uploaded documents.</div>
</div>
<div class="di-card">
<div class="di-card-icon">🧩</div>
<div class="di-card-title">Solve &amp; Reason</div>
<div class="di-card-text">Apply formulas, methods, and concepts from your documents to new numerical problems and scenarios.</div>
</div>
<div class="di-card">
<div class="di-card-icon">📝</div>
<div class="di-card-title">Smart Summaries</div>
<div class="di-card-text">Turn long PDFs into structured summaries containing key concepts, important findings, and conclusions.</div>
</div>
<div class="di-card">
<div class="di-card-icon">🎯</div>
<div class="di-card-title">Interactive Practice</div>
<div class="di-card-text">Generate MCQs, numerical problems, conceptual questions, and case studies directly from your documents.</div>
</div>
</div>
<div class="di-cta">
<strong>Ready to learn?</strong><br>
Upload your PDF(s) from the sidebar to open your AI-powered document workspace.
</div>
</div>"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )
# ==========================================================
# Show PDF Summary
# ==========================================================

if st.session_state.summary:
    with st.expander("📝 PDF Summary", expanded=True):
        st.markdown(st.session_state.summary)


# ==========================================================
# Build Vector Store
# ==========================================================

if uploaded_files:

    # A stable signature lets Streamlit know when the selected PDF collection changes.
    upload_signature = tuple(
        sorted((file.name, file.size) for file in uploaded_files)
    )

    # Only rebuild when the uploaded PDF collection changes.
    if (
        "vectorstore" not in st.session_state
        or st.session_state.get("upload_signature") != upload_signature
    ):

        all_documents = []
        page_images = {}

        with st.spinner("📄 Reading PDFs..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    pdf_path = tmp.name

                loader = PyPDFLoader(pdf_path)
                file_documents = loader.load()

                # Preserve the source PDF name on every page/chunk.
                for document in file_documents:
                    document.metadata["source_file"] = uploaded_file.name

                all_documents.extend(file_documents)

                # Generate page preview images for this PDF.
                pdf = fitz.open(pdf_path)
                file_page_images = []

                for page in pdf:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    safe_name = os.path.basename(uploaded_file.name).replace(" ", "_")
                    image_path = os.path.join(
                        tempfile.gettempdir(),
                        f"{safe_name}_page_{page.number}.png"
                    )
                    pix.save(image_path)
                    file_page_images.append(image_path)

                pdf.close()
                page_images[uploaded_file.name] = file_page_images

                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

        with st.spinner("✂ Splitting documents..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(all_documents)

        with st.spinner("🧠 Creating embeddings..."):
            vectorstore = FAISS.from_documents(
                chunks,
                embedding_model
            )

        st.session_state.vectorstore = vectorstore
        st.session_state.upload_signature = upload_signature
        st.session_state.filenames = [file.name for file in uploaded_files]
        st.session_state.documents = all_documents
        st.session_state.chunks = chunks
        st.session_state.page_images = page_images
        st.session_state.summary = ""
        st.session_state.generate_summary = False
        st.session_state.practice_questions = ""
        st.session_state.practice_answers = ""
        st.session_state.show_practice_answers = False
        st.session_state.generate_practice = False
        st.session_state.practice_data = []
        st.session_state.practice_index = 0
        st.session_state.practice_results = []
        st.session_state.practice_submitted = False

        # ==========================================================
        # Generate Suggested Questions
        # ==========================================================

        sample_context = ""

        for chunk in chunks[:3]:
            sample_context += chunk.page_content
            sample_context += "\n\n"

        suggestion_prompt = f"""
You are reading one or more PDFs.

Based ONLY on the content below, generate exactly 5 short questions
that a student might ask.

Return ONLY the questions.

Context:

{sample_context}
"""

        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=suggestion_prompt
            )

            questions = [
                q.strip("-•1234567890. ")
                for q in response.text.split("\n")
                if q.strip()
            ]

            st.session_state.suggested_questions = questions[:5]

        except Exception as e:

            st.session_state.suggested_questions = []
            st.warning(f"Suggested questions could not be generated: {e}")

    vectorstore = st.session_state.vectorstore

    # ==========================================================
    # Generate PDF Summary only when the button is clicked
    # ==========================================================

    if st.session_state.generate_summary and "chunks" in st.session_state:
        with st.spinner("📝 Generating PDF Summary..."):
            pdf_text = ""

            for chunk in st.session_state.chunks[:8]:
                pdf_text += chunk.page_content
                pdf_text += "\n\n"

            summary_prompt = f"""
Summarize the following PDF collection as one combined summary.

Include:
• Main Topic
• Key Concepts
• Important Findings
• Conclusion

Document:

{pdf_text}
"""

            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=summary_prompt
                )
                st.session_state.summary = response.text
            except Exception as e:
                st.error(f"⚠️ Unable to generate summary: {e}")
            finally:
                st.session_state.generate_summary = False

            st.rerun()

    # ==========================================================
    # Interactive Quiz / Practice Generator
    # ==========================================================

    if st.session_state.generate_practice and "chunks" in st.session_state:
        with st.spinner("🧠 Creating an interactive practice set from your PDFs..."):

            practice_context = ""
            sampled_chunks = st.session_state.chunks[:20]

            for chunk in sampled_chunks:
                source_file = chunk.metadata.get("source_file", "PDF")
                page_number = chunk.metadata.get("page", 0) + 1
                practice_context += (
                    f"Source: {source_file}, Page {page_number}\n"
                    f"{chunk.page_content}\n\n"
                )

            practice_prompt = f"""
You are a document-grounded tutor creating a NEW interactive practice set.

Use ONLY concepts, formulas, methods, facts, and frameworks supported by the
PDF context below. Do not copy an existing question verbatim.

Create exactly {st.session_state.practice_count} questions.
Requested Question Type: {st.session_state.practice_type}
Difficulty: {st.session_state.practice_difficulty}

Return ONLY valid JSON. No markdown fences and no text before or after JSON.

JSON schema:
{{
  "questions": [
    {{
      "type": "MCQ" | "Numerical" | "Conceptual" | "Case Study",
      "question": "question text",
      "options": ["A option", "B option", "C option", "D option"],
      "correct_answer": "exact correct option text for MCQ, otherwise concise expected answer",
      "explanation": "solution or explanation grounded in the PDF concept",
      "source_file": "exact PDF filename",
      "source_page": 1
    }}
  ]
}}

Rules:
- For MCQ, options MUST contain exactly 4 choices and correct_answer must exactly
  match one item in options.
- For Numerical, Conceptual, and Case Study, options MUST be [].
- Numerical questions must use NEW values but document-supported methods/formulas.
- Case studies must be NEW scenarios requiring application of PDF-supported concepts.
- Mixed should contain a useful mixture of question types.
- Every question must be answerable from the supplied PDF-supported knowledge.
- source_file and source_page must identify the concept used.

PDF Context:

{practice_context}
"""

            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=practice_prompt
                )

                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = raw.replace("```json", "", 1).replace("```", "").strip()

                parsed = json.loads(raw)
                questions = parsed.get("questions", [])

                if not questions:
                    raise ValueError("Gemini returned no practice questions.")

                st.session_state.practice_data = questions[:st.session_state.practice_count]
                st.session_state.practice_index = 0
                st.session_state.practice_results = []
                st.session_state.practice_submitted = False
                st.session_state.practice_questions = "interactive"
                st.session_state.practice_answers = ""

            except Exception as e:
                st.error(f"⚠️ Unable to generate interactive practice: {e}")

            finally:
                st.session_state.generate_practice = False

            st.rerun()

    # ==========================================================
    # Interactive Practice Area
    # ==========================================================

    if st.session_state.practice_data:
        questions = st.session_state.practice_data
        index = st.session_state.practice_index
        total = len(questions)

        st.markdown("---")
        st.markdown("## 🧠 Interactive Quiz / Practice")

        # Finished quiz
        if index >= total:
            correct_count = sum(
                1 for result in st.session_state.practice_results
                if result.get("correct")
            )
            accuracy = round((correct_count / total) * 100) if total else 0

            st.markdown("### 🎯 Quiz Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{correct_count} / {total}")
            col2.metric("Accuracy", f"{accuracy}%")
            col3.metric("Incorrect", total - correct_count)

            with st.expander("📋 Review Answers", expanded=True):
                for i, result in enumerate(st.session_state.practice_results, start=1):
                    icon = "✅" if result.get("correct") else "❌"
                    st.markdown(f"#### {icon} Question {i}")
                    st.write(result.get("question", ""))
                    st.write(f"**Your Answer:** {result.get('user_answer', '')}")
                    st.write(f"**Correct Answer:** {result.get('correct_answer', '')}")
                    st.write(f"**Explanation:** {result.get('explanation', '')}")
                    st.caption(
                        f"📚 Concept Source: {result.get('source_file', 'PDF')} "
                        f"— Page {result.get('source_page', 1)}"
                    )
                    if i < len(st.session_state.practice_results):
                        st.markdown("---")

            if st.button("🔄 Generate New Quiz", use_container_width=True):
                st.session_state.generate_practice = True
                st.session_state.practice_data = []
                st.session_state.practice_results = []
                st.session_state.practice_index = 0
                st.session_state.practice_submitted = False
                st.rerun()

        else:
            q = questions[index]
            qtype = q.get("type", "Conceptual")

            st.caption(
                f"Question {index + 1} of {total} • "
                f"{qtype} • {st.session_state.practice_difficulty}"
            )
            st.progress((index + 1) / total)
            st.markdown(f"### {q.get('question', '')}")

            answer_key = f"practice_answer_{index}"

            if qtype == "MCQ" and q.get("options"):
                user_answer = st.radio(
                    "Choose your answer:",
                    q["options"],
                    key=answer_key,
                    index=None
                )
            else:
                user_answer = st.text_area(
                    "Enter your answer / working:",
                    key=answer_key,
                    height=140,
                    placeholder="Type your answer here..."
                )

            if not st.session_state.practice_submitted:
                if st.button("✅ Submit Answer", use_container_width=True):
                    if not user_answer or not str(user_answer).strip():
                        st.warning("Please enter or select an answer first.")
                    else:
                        # MCQ can be graded locally. Open responses are evaluated
                        # against the document-grounded expected answer.
                        if qtype == "MCQ":
                            is_correct = (
                                str(user_answer).strip()
                                == str(q.get("correct_answer", "")).strip()
                            )
                            feedback = q.get("explanation", "")
                        else:
                            grading_prompt = f"""
You are grading a student's answer using a document-grounded expected answer.

Question:
{q.get("question", "")}

Expected Answer:
{q.get("correct_answer", "")}

Expected Explanation:
{q.get("explanation", "")}

Student Answer:
{user_answer}

Decide whether the student's answer is substantively correct.
For numerical answers, allow equivalent working/formatting.
For conceptual/case-study answers, allow different wording if the core reasoning
matches the expected document-grounded answer.

Return ONLY valid JSON:
{{
  "correct": true,
  "feedback": "brief specific feedback"
}}
"""
                            try:
                                grade_response = client.models.generate_content(
                                    model="gemini-flash-latest",
                                    contents=grading_prompt
                                )
                                grade_raw = grade_response.text.strip()
                                if grade_raw.startswith("```"):
                                    grade_raw = grade_raw.replace(
                                        "```json", "", 1
                                    ).replace("```", "").strip()
                                grade = json.loads(grade_raw)
                                is_correct = bool(grade.get("correct", False))
                                feedback = grade.get(
                                    "feedback",
                                    q.get("explanation", "")
                                )
                            except Exception:
                                is_correct = False
                                feedback = (
                                    "Automatic grading was unavailable. "
                                    "Compare your answer with the solution below."
                                )

                        st.session_state.practice_results.append(
                            {
                                "question": q.get("question", ""),
                                "user_answer": str(user_answer),
                                "correct": is_correct,
                                "correct_answer": q.get("correct_answer", ""),
                                "explanation": q.get("explanation", ""),
                                "feedback": feedback,
                                "source_file": q.get("source_file", "PDF"),
                                "source_page": q.get("source_page", 1),
                            }
                        )
                        st.session_state.practice_submitted = True
                        st.rerun()

            else:
                result = st.session_state.practice_results[-1]

                if result.get("correct"):
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Not quite.")

                if result.get("feedback"):
                    st.write(f"**Feedback:** {result['feedback']}")

                st.write(f"**Correct Answer:** {q.get('correct_answer', '')}")
                st.write(f"**Explanation:** {q.get('explanation', '')}")
                st.caption(
                    f"📚 Concept Source: {q.get('source_file', 'PDF')} "
                    f"— Page {q.get('source_page', 1)}"
                )

                if st.button(
                    "Next Question →" if index + 1 < total else "View Results →",
                    use_container_width=True
                ):
                    st.session_state.practice_index += 1
                    st.session_state.practice_submitted = False
                    st.rerun()

        st.markdown("---")

    # ==========================================================
    # Sidebar Information
    # ==========================================================

    st.sidebar.success(f"✅ {len(st.session_state.filenames)} PDF(s) Loaded")

    st.sidebar.write(
        f"📚 Documents : {len(st.session_state.filenames)}"
    )

    st.sidebar.write(
        f"📄 Total Pages : {len(st.session_state.documents)}"
    )

    st.sidebar.write(
        f"🧩 Total Chunks : {len(st.session_state.chunks)}"
    )

    st.sidebar.write("🤖 Embedding Model")

    st.sidebar.code("all-MiniLM-L6-v2")

    # ==========================================================
    # Show Previous Chat
    # ==========================================================

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            # Show sources grouped by PDF for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                render_grouped_sources(msg["sources"])

    # ==========================================================
    # Chat Input
    # ==========================================================
# ==========================================================
# Suggested Questions
# ==========================================================

    if st.session_state.suggested_questions:
     

     st.markdown("### 💡 Suggested Questions")
     st.caption("Start with one of these document-grounded questions.")

     for i, q in enumerate(st.session_state.suggested_questions):


        if st.button(
            q,
            key=f"suggestion_{i}",
            use_container_width=True
        ):
            st.session_state.selected_question = q
            st.rerun()

    st.markdown("---")

# ==========================================================
# Chat Input
# ==========================================================

typed_question = st.chat_input(
    "Ask something about your PDF..."
)

question = typed_question

if question is None:
    question = st.session_state.selected_question
    st.session_state.selected_question = None

if question:
    st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

    with st.chat_message("user"):

            st.markdown(question)

    with st.spinner("🔍 Searching relevant pages..."):

            docs_and_scores = vectorstore.similarity_search_with_score(
                question,
                k=3
            )

            context = ""

            docs = []
            scores = []

            for doc, score in docs_and_scores:

                docs.append(doc)
                scores.append(score)

                source_file = doc.metadata.get("source_file", "PDF")
                page_number = doc.metadata.get("page", 0) + 1
                context += f"Source: {source_file}, Page {page_number}\n"
                context += doc.page_content
                context += "\n\n"

            best_score = scores[0]

            confidence = max(
                0,
                min(
                    100,
                    int(100 - best_score)
                )
            )
            conversation_history = ""

            for msg in st.session_state.messages[-6:]:

                conversation_history += (
                    f"{msg['role'].capitalize()}: "
                    f"{msg['content']}\n"
                )
        

            prompt = f"""
You are a helpful document-grounded AI tutor.

Use the previous conversation if the user asks a follow-up question.

The uploaded document context is your PRIMARY knowledge source.

Follow these rules carefully:

1. DIRECT DOCUMENT QUESTIONS
If the user asks for a definition, fact, formula, explanation, topic, or other
information stated in the document, answer using the document context.

2. NUMERICAL / APPLICATION QUESTIONS
If the user gives a NEW numerical problem or scenario, the exact problem does
NOT need to appear in the document.
Use the formulas, concepts, methods, rules, or principles found in the document
context and apply them to the values or scenario supplied by the user.
Show the important calculation steps clearly and give the final answer.

3. CASE STUDIES / REASONING QUESTIONS
If the user provides a new case study or situation, apply the relevant concepts,
frameworks, methods, or principles found in the document context to analyze it.
Clearly explain how the document concepts apply to the new case.

4. GROUNDING
Do not invent document facts, formulas, concepts, or methods that are not
supported by the retrieved document context.
You may perform arithmetic, algebra, logical reasoning, comparison, deduction,
and application necessary to use the document-supported concepts.

5. INSUFFICIENT DOCUMENT KNOWLEDGE
If neither the answer nor the concepts/methods needed to derive the answer are
supported by the document context, reply exactly:

I couldn't find enough information in the document to answer this question.

6. ANSWER STYLE
For a direct question, answer naturally and concisely.
For a numerical problem, show:
- Formula/Concept Used
- Calculation/Steps
- Final Answer
For a case study, show:
- Relevant Document Concepts
- Application to the Case
- Conclusion/Recommendation

================================================

Previous Conversation:

{conversation_history}

================================================

Document Context:

{context}

================================================

Current Question:

{question}

Answer:
"""

    with st.spinner("🤖 Gemini is generating the answer..."):

            try:

                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt
                )

                answer = response.text

            except Exception as e:

                answer = (
                    "⚠️ Unable to generate an answer because the Gemini API quota "
                    "has been exceeded. Please try again later or use another API key."
                )

                st.error(answer)

    with st.chat_message("assistant"):

            if confidence >= 85:

                st.success(f"🟢 Confidence : {confidence}%")

            elif confidence >= 70:

                st.warning(f"🟡 Confidence : {confidence}%")

            else:

                st.error(f"🔴 Confidence : {confidence}%")

            st.markdown(answer)
            st.expander("📋 Copy Answer").code(answer,language=None)

            # Show referenced pages grouped by PDF
            reference_groups = {}
            for doc in docs:
                source_file = doc.metadata.get("source_file", "PDF")
                page_number = doc.metadata.get("page", 0) + 1
                reference_groups.setdefault(source_file, set()).add(page_number)

            st.markdown("📖 **Referenced Documents**")
            for source_file, page_numbers in reference_groups.items():
                pages_text = ", ".join(str(p) for p in sorted(page_numbers))
                page_word = "Page" if len(page_numbers) == 1 else "Pages"
                st.caption(f"• {source_file} — {page_word} {pages_text}")

            current_sources = [
                {
                    "file": doc.metadata.get("source_file", "PDF"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "confidence": max(0, min(100, int(100 - score))),
                    "preview": doc.page_content[:250],
                    "text": doc.page_content
                }
                for doc, score in docs_and_scores
            ]

            render_grouped_sources(current_sources)

    st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "confidence": confidence,
                "sources": [
                    {
                        "file": doc.metadata.get("source_file", "PDF"),
                        "page": doc.metadata.get("page", 0) + 1,
                        "confidence": max(0, min(100, int(100 - score))),
                        "preview": doc.page_content[:250],
                        "text": doc.page_content
                    }
                    for doc, score in docs_and_scores
                ]
            }
        )

# ==========================================================
# Download Conversation
# ==========================================================

if st.session_state.messages or st.session_state.summary:

    chat_text = ""

    if st.session_state.summary:
        chat_text += "PDF SUMMARY\n"
        chat_text += "=" * 60 + "\n\n"
        chat_text += st.session_state.summary + "\n\n"
        chat_text += "=" * 60 + "\n"
        chat_text += "CONVERSATION\n"
        chat_text += "=" * 60 + "\n\n"

    for msg in st.session_state.messages:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")

        chat_text += f"{role}:\n"
        chat_text += content + "\n\n"
        chat_text += "-" * 60 + "\n\n"

    st.download_button(
        "⬇ Download Conversation",
        data=chat_text,
        file_name="conversation.txt",
        mime="text/plain",
        use_container_width=True
    )
