import os
import json
import tempfile
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

st.set_page_config(
    page_title="DocInsight AI",
    page_icon="📄",
    layout="wide"
)

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
# Theme State
# ==========================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ==========================================================
# Sidebar
# ==========================================================
with st.sidebar:

    # ======================================================
    # Branding
    # ======================================================
    st.markdown("""
    <div style="padding: 8px 0 5px 0;">
        <h2 style="margin-bottom: 2px;">📄 DocInsight AI</h2>
        <p style="
            color: #6b7280;
            font-size: 13px;
            margin-top: 0;
        ">
            Your AI document learning assistant
        </p>
    </div>
    """, unsafe_allow_html=True)

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
    # Appearance
    # ======================================================
    st.markdown("### 🎨 Appearance")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_mode_toggle"
    )
    st.session_state.dark_mode = dark_mode

    st.markdown("---")

    # ======================================================
    # Conversation
    # ======================================================
    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Gemini • FAISS • HuggingFace")


# ==========================================================
# App Theme Styling
# ==========================================================

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #f3f4f6; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #e5e7eb !important; }
    h1, h2, h3, h4 { color: #f8fafc !important; }
    p, label { color: #d1d5db; }
    .stButton > button, .stDownloadButton > button { background-color: #21262d; color: #f8fafc; border: 1px solid #30363d; border-radius: 10px; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #8b5cf6; color: #ffffff; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stTextArea"] textarea { background-color: #21262d !important; color: #f8fafc !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #21262d; border-color: #30363d; }
    [data-testid="stChatInput"] { background-color: #161b22; border-color: #30363d; }
    [data-testid="stChatInput"] textarea { color: #f8fafc !important; }
    [data-testid="stExpander"] { background-color: #161b22; border-color: #30363d; border-radius: 10px; }
    hr { border-color: #30363d; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# Main Page
# ==========================================================
st.markdown("""
<div style="padding: 12px 0 20px 0;">
    <h1 style="margin-bottom: 4px;">📄 DocInsight AI</h1>
    <p style="
        font-size: 19px;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 8px;
    ">
        Turn your documents into answers, insights, and practice.
    </p>
    <p style="
        font-size: 13px;
        color: #9ca3af;
        margin-top: 0;
    ">
        AI-powered document learning • Multi-PDF Q&A • Summaries • Practice Generator
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# Session State
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "generate_summary" not in st.session_state:
    st.session_state.generate_summary = False

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "generate_practice" not in st.session_state:
    st.session_state.generate_practice = False

if "practice_questions" not in st.session_state:
    st.session_state.practice_questions = ""

if "practice_answers" not in st.session_state:
    st.session_state.practice_answers = ""

if "show_practice_answers" not in st.session_state:
    st.session_state.show_practice_answers = False

if "practice_data" not in st.session_state:
    st.session_state.practice_data = []

if "practice_index" not in st.session_state:
    st.session_state.practice_index = 0

if "practice_results" not in st.session_state:
    st.session_state.practice_results = []

if "practice_submitted" not in st.session_state:
    st.session_state.practice_submitted = False

# ==========================================================
# Show PDF Summary
# ==========================================================

if st.session_state.summary:
    with st.expander("📝 PDF Summary", expanded=True):
        st.markdown(st.session_state.summary)

# ==========================================================
# Download Conversation
# ==========================================================

chat_text = ""

# Include the generated PDF summary in the downloaded conversation file.
if st.session_state.summary:
    chat_text += "PDF SUMMARY\n"
    chat_text += "=" * 60
    chat_text += "\n\n"
    chat_text += st.session_state.summary
    chat_text += "\n\n"
    chat_text += "=" * 60
    chat_text += "\nCONVERSATION\n"
    chat_text += "=" * 60
    chat_text += "\n\n"

for msg in st.session_state.messages:

    chat_text += f"{msg['role'].capitalize()}:\n"

    chat_text += msg["content"]

    chat_text += "\n\n"

    chat_text += "-" * 60

    chat_text += "\n\n"
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

if st.session_state.messages or st.session_state.summary:

    st.download_button(
        "⬇ Download Conversation",
        data=chat_text,
        file_name="conversation.txt",
        mime="text/plain"
    )

else:

    st.info("👈 Upload one or more PDFs from the sidebar to start chatting.")