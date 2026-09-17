import inspect
import streamlit as st
from src.document_loader import extract_text
from src.chunking import split_documents
from src.embeddings import generate_embeddings
from src.vector_store import store_embeddings
from src.rag_pipeline import retrieve_documents
from src.scope_agent import analyze_scope
from src.risk_agent import analyze_risks
from src.documentation_agent import generate_project_documentation
from src.health_agent import analyze_project_health
from src.blocker_agent import analyze_blockers
from src.llm import generate_answer




st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Interactive module cards */
.module-card {
    position: relative;
    overflow: hidden;
    border: 1px solid #38445c;
    border-left: 6px solid var(--accent, #7c9cff);
    border-radius: 18px;
    padding: 22px 24px;
    margin: 22px 0 10px 0;
    background: linear-gradient(135deg, #171d2b 0%, #20283a 100%);
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.module-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(90,120,255,.20);
    border-color: #9cafff;
}
.module-card:before {
    content: "";
    position: absolute;
    width: 130px;
    height: 130px;
    right: -55px;
    top: -65px;
    border-radius: 50%;
    background: rgba(124,156,255,.08);
}
.module-icon { font-size: 34px; line-height: 1; margin-bottom: 10px; }
.module-title { color: #ffffff; font-size: 27px; font-weight: 800; margin: 0 0 8px 0; }
.module-description { color: #b2bdd0; font-size: 15px; line-height: 1.65; margin: 0 0 13px 0; }
.module-badge {
    display: inline-block;
    color: #c5d2ff;
    background: #263653;
    border: 1px solid #526da8;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 700;
}
.module-line { height: 6px; background: #30394d; border-radius: 99px; margin-top: 16px; }
.module-line span { display:block; height:100%; width: var(--progress,70%); border-radius:99px; background: linear-gradient(90deg,var(--accent,#7c9cff),#b58cff); }
.module-caption { color:#8794aa; font-size:12px; margin-top:8px; }
div.stButton > button {
    border-radius: 12px;
    border: 1px solid #53617a;
    font-weight: 700;
    transition: all .2s ease;
}
div.stButton > button:hover {
    border-color: #9cafff;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(124,156,255,.18);
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>

.section-card {
    background: linear-gradient(135deg, #171d2b, #20283a);
    border: 1px solid #39445c;
    border-left: 6px solid #7c9cff;
    border-radius: 18px;
    padding: 24px;
    margin: 18px 0;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.20);
    transition: all 0.25s ease;
}

.section-card:hover {
    transform: translateY(-4px);
    border-color: #9cafff;
    box-shadow: 0 12px 32px rgba(90, 120, 255, 0.22);
}

.section-icon {
    font-size: 34px;
    margin-bottom: 8px;
}

.section-title {
    color: #ffffff;
    font-size: 25px;
    font-weight: 750;
    margin-bottom: 8px;
}

.section-description {
    color: #aeb9ce;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 15px;
}

.status-badge {
    display: inline-block;
    background: #263653;
    color: #b9caff;
    border: 1px solid #526da8;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

.progress-container {
    background: #30394d;
    border-radius: 20px;
    height: 8px;
    width: 100%;
    margin-top: 12px;
}

.progress-fill {
    background: linear-gradient(90deg, #7c9cff, #b58cff);
    height: 8px;
    border-radius: 20px;
}

.small-label {
    color: #8f9bb2;
    font-size: 12px;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 15px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="subtitle">'
    'AI-powered project monitoring, risk analysis, '
    'and decision support'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* Main application */
    .stApp {
        background-color: #0e1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #151922;
        border-right: 1px solid #262b36;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #8b93a7;
        margin-bottom: 25px;
    }

    /* Main heading */
    .main-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #9aa3b5;
        font-size: 16px;
        margin-bottom: 30px;
    }

   
    /* Cards */

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
}

.metric-card.risk-card {
    border-left: 4px solid #f4a261;
    background: rgba(244, 162, 97, 0.08);
}

.metric-card.blocker-card {
    border-left: 4px solid #e9c46a;
    background: rgba(233, 196, 106, 0.08);
}

.metric-card.health-card {
    border-left: 4px solid #8abf8a;
    background: rgba(138, 191, 138, 0.08);
}

.metric-card.document-card {
    border-left: 4px solid #7aa2f7;
    background: rgba(122, 162, 247, 0.08);
}

.metric-title {
    color: #8f98aa;
    font-size: 14px;
    margin-bottom: 10px;
}

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    /* Section */
    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #8f98aa;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Upload box */
    .upload-container {
        background-color: #151922;
        border: 1px solid #272d39;
        border-radius: 12px;
        padding: 20px;
    }

    /* RAG pipeline */
    .pipeline-container {
        background-color: #151922;
        border: 1px solid #272d39;
        border-radius: 12px;
        padding: 25px;
        margin-top: 10px;
    }

    .pipeline-step {
        text-align: center;
        padding: 15px;
        background-color: #1b202b;
        border-radius: 10px;
        border: 1px solid #2a303d;
    }

    .pipeline-name {
        font-weight: 600;
        font-size: 14px;
    }

    .pipeline-status {
        color: #7d8799;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Info box */
    .info-box {
        background-color: #151922;
        border: 1px solid #272d39;
        border-radius: 12px;
        padding: 20px;
        color: #aeb6c5;
        line-height: 1.6;
    }
    
    .status-badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-badge.completed {
    color: #8abf8a;
    background: rgba(138, 191, 138, 0.12);
    border: 1px solid rgba(138, 191, 138, 0.25);
}

.status-badge.waiting {
    color: #8f98aa;
    background: rgba(143, 152, 170, 0.08);
    border: 1px solid rgba(143, 152, 170, 0.20);
}

    

</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
import streamlit as st

# ---------------- SIDEBAR DESIGN ----------------
st.markdown(
    """
    <style>
    /* Sidebar width */
    section[data-testid="stSidebar"] {
        width: 290px !important;
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
        border-right: 1px solid #263244;
    }

    /* Sidebar padding */
    section[data-testid="stSidebar"] > div {
        padding: 28px 18px;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 28px;
    }

    .sidebar-line {
        height: 1px;
        background: #334155;
        margin: 18px 0 24px 0;
    }

    /* Radio button labels */
    section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 9px 12px !important;
        border-radius: 10px !important;
        margin-bottom: 5px !important;
        transition: all 0.2s ease-in-out;
    }

    section[data-testid="stSidebar"] label:hover {
        background: #1e293b !important;
        color: #ffffff !important;
    }

    /* Hide radio circles */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Radio group spacing */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 7px !important;
    }

    /* Selected navigation item */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        box-shadow: 0 5px 14px rgba(37, 99, 235, 0.25);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar header
st.sidebar.markdown(
    """
    <div class="sidebar-title">AI Project Advisor</div>
    <div class="sidebar-subtitle">Project Intelligence Platform</div>
    <div class="sidebar-line"></div>
    """,
    unsafe_allow_html=True
)

# Navigation menu
menu_items = [
    "Dashboard",
    "Documents",
    "RAG Pipeline",
    "Project Scope",
    "Risks & Forecast",
    "Blockers & Actions",
    "Documentation",
    "Project Health",
    "AI Assistant",
]

selected_page = st.sidebar.radio(
    "Navigation",
    menu_items,
    label_visibility="collapsed"
)
   

# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">AI Project Intelligence & Risk Advisor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Transform project documents into actionable project intelligence.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []

if "risk_analysis" not in st.session_state:
    st.session_state.risk_analysis = ""

if "blocker_analysis" not in st.session_state:
    st.session_state.blocker_analysis = ""

if "health_analysis" not in st.session_state:
    st.session_state.health_analysis = ""
# ---------------------------------------------------------
# DASHBOARD METRICS
# ---------------------------------------------------------

documents_count = len(
    st.session_state.get("processed_documents", [])
)

risk_analysis = st.session_state.get("risk_analysis", "")

if risk_analysis:
    risks_count = risk_analysis.lower().count("risk")
else:
    risks_count = 0

blocker_analysis = st.session_state.get("blocker_analysis", "")

if blocker_analysis:
    blockers_count = blocker_analysis.lower().count("blocker")
else:
    blockers_count = 0

health_analysis = st.session_state.get("health_analysis", "")

if health_analysis:
    health_text = health_analysis.lower()

    if "critical" in health_text:
        project_health = "Critical"
    elif "at risk" in health_text:
        project_health = "At Risk"
    elif "healthy" in health_text:
        project_health = "Healthy"
    else:
        project_health = "Analyzed"
else:
    project_health = "--"


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card document-card">
        <div class="metric-title">Documents</div>
        <div class="metric-value">{documents_count}</div>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown(f"""
    <div class="metric-card risk-card">
        <div class="metric-title">Identified Risks</div>
        <div class="metric-value">{risks_count}</div>
    </div>
    """, unsafe_allow_html=True)


with col3:
    st.markdown(f"""
    <div class="metric-card blocker-card">
        <div class="metric-title">Blockers</div>
        <div class="metric-value">{blockers_count}</div>
    </div>
    """, unsafe_allow_html=True)


with col4:
    st.markdown(f"""
    <div class="metric-card health-card">
        <div class="metric-title">Project Health</div>
        <div class="metric-value">{project_health}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# DOCUMENT UPLOAD
# ---------------------------------------------------------
st.markdown("""
<div class="module-card" style="--accent:#67d5ff;--progress:45%;">
  <div class="module-icon">📁</div>
  <div class="module-title">Upload Project Documents</div>
  <span class="module-badge">Knowledge Base</span>
  <div class="module-description">Upload multiple project artifacts to build the project knowledge base.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">PDF • DOCX • CSV • TXT</div>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="section-description">'
    'Upload multiple project artifacts to build the project knowledge base.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Supported formats: PDF, DOCX, CSV, TXT",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True,
    label_visibility="visible"
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} document(s) selected.")

    for file in uploaded_files:

        st.write(
            f"**{file.name}** — {file.size / 1024:.1f} KB"
        )

    st.divider()

    if st.button("Process Documents", type="primary"):

        with st.spinner("Extracting text from documents..."):

            st.session_state.processed_documents = []

            processed_documents = st.session_state.processed_documents

            for file in uploaded_files:

                try:
                    text = extract_text(file)

                    processed_documents.append({
                        "filename": file.name,
                        "file_type": file.name.split(".")[-1].upper(),
                        "text": text
                    })

                except Exception as e:

                    st.error(
                        f"Could not process {file.name}: {e}"
                    )

        if processed_documents:

            st.success(
                f"Successfully processed "
                f"{len(processed_documents)} document(s)."
            )

            st.session_state.processed_documents = processed_documents

            for document in processed_documents:

                with st.expander(
                    f"{document['filename']} "
                    f"({document['file_type']})"
                ):

                    st.write(
                        f"Characters extracted: "
                        f"{len(document['text'])}"
                    )

                    st.text_area(
                        "Extracted Text",
                        document["text"][:3000],
                        height=200,
                        key=document["filename"]
                    )
                    # Create chunks from extracted documents
            chunks = split_documents(processed_documents)
            st.session_state.chunks = chunks

            st.info(
                f"Created {len(chunks)} text chunks from "
                f"{len(processed_documents)} documents."
            )

            with st.spinner("Generating embeddings..."):

                embeddings = generate_embeddings(chunks)
                st.session_state.embeddings = embeddings

            st.success(
                f"Generated embeddings for {len(embeddings)} chunks."
            )

            with st.spinner("Indexing embeddings in ChromaDB..."):

                indexed_count = store_embeddings(
                chunks,
                embeddings
            )
            st.session_state.indexed_count = indexed_count

            st.success(
                f"Indexed {indexed_count} chunks in ChromaDB."
            )
# ---------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------
st.markdown("""
<div class="module-card" style="--accent:#7c9cff;--progress:70%;">
  <div class="module-icon">🔗</div>
  <div class="module-title">RAG Pipeline</div>
  <span class="module-badge">Knowledge Processing</span>
  <div class="module-description">Track how uploaded documents move through extraction, chunking, embeddings, and indexing.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Upload → Extract → Chunk → Embed → Index</div>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="section-description">'
    'Documents will pass through the following knowledge processing pipeline.'
    '</div>',
    unsafe_allow_html=True
)

pipeline_cols = st.columns(5)

processed_documents = st.session_state.get(
    "processed_documents", []
)

if processed_documents:
    upload_status = "Completed"
    extract_status = "Completed"
else:
    upload_status = "Waiting"
    extract_status = "Waiting"

chunks = st.session_state.get("chunks", [])

if len(chunks) > 0:
    chunk_status = "Completed"
else:
    chunk_status = "Waiting"

embeddings = st.session_state.get("embeddings")

if embeddings is not None and len(embeddings) > 0:
    embed_status = "Completed"
else:
    embed_status = "Waiting"

if st.session_state.get("indexed_count", 0) > 0:
    index_status = "Completed"
else:
    index_status = "Waiting"


pipeline_steps = [
    ("01", "Upload", upload_status),
    ("02", "Extract", extract_status),
    ("03", "Chunk", chunk_status),
    ("04", "Embed", embed_status),
    ("05", "Index", index_status)
]

for col, (number, name, status) in zip(pipeline_cols, pipeline_steps):

    with col:

        st.markdown(
            f"""
            <div class="pipeline-step">
                <div style="font-size:12px;color:#7d8799;">
                    STEP {number}
                </div>
                <div class="pipeline-name">
                    {name}
                </div>
                <div class="pipeline-status">
    <span class="status-badge {status.lower()}">
        {status}
    </span>
</div>
            </div>
            """,
            unsafe_allow_html=True
        )
# ---------------------------------------------------------
# RAG RETRIEVAL TEST
# ---------------------------------------------------------

st.markdown("""
<div class="module-card" style="--accent:#67d5ff;--progress:92%;">
  <div class="module-icon">🤖</div>
  <div class="module-title">AI Project Assistant</div>
  <span class="module-badge">Ask Your Project</span>
  <div class="module-description">Ask questions about uploaded project documents, risks, blockers, milestones, and progress.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Ask questions • Retrieve sources • Generate answers</div>
</div>
""", unsafe_allow_html=True)


query = st.text_input(
    "Ask a question about your uploaded project documents",
    placeholder="Example: What are the current project blockers?"
)

if query:

    results = retrieve_documents(query)

    if results:
        st.success(
            f"Retrieved {len(results)} relevant document chunks"
        )

        with st.spinner("Generating AI answer..."):
                context = "\n\n".join(
                result["text"] for result in results
            )

                prompt = f"""
You are an AI Project Intelligence Assistant.

Answer the user's question using only the project document context below.

Project Document Context:
{context}

User Question:
{query}

Give a clear, practical, and professional answer.
"""

        answer = generate_answer(prompt)

        st.markdown("### AI Answer")
        st.write(answer)

        st.markdown("### Retrieved Sources")

        for result in results:
            st.write(
                f"**{result['filename']}**"
            )

            st.caption(result["text"])

            st.caption(
                f"Source: {result['filename']} | "
                f"Distance: {result['distance']:.4f}"
            )

    else:
        st.warning(
            "No relevant information found in the project knowledge base."
        )

st.markdown("---")

st.markdown("""
<div class="module-card" style="--accent:#7c9cff;--progress:72%;">
  <div class="module-icon">🎯</div>
  <div class="module-title">Project Scope & Deliverables</div>
  <span class="module-badge">Scope Intelligence</span>
  <div class="module-description">Analyze project documents to identify the project objective, scope, modules, deliverables, requirements, and milestones.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Objective • Modules • Deliverables • Milestones</div>
</div>
""", unsafe_allow_html=True)


if st.button("Analyze Project Scope"):

    scope_chunks = retrieve_documents(
        "project objective scope modules features "
        "deliverables requirements deadlines milestones",
        top_k=6
    )

    if scope_chunks:

        with st.spinner("Analyzing project scope..."):

            scope_analysis = analyze_scope(
                scope_chunks
            )
            st.session_state.scope_analysis = scope_analysis

        st.success(
            "Project scope analysis completed."
        )

        

    else:

        st.warning(
            "No relevant project information found. "
            "Please process your project documents first."
        )

if "scope_analysis" in st.session_state:

    st.markdown("### Project Scope")

    with st.expander("View Complete Scope Analysis", expanded=True):
        st.write(st.session_state.scope_analysis)

st.markdown("### Important Deadlines & Milestones")

milestones = [
    {
        "name": "Sprint 3 Completion",
        "status": "Blocked",
        "progress": 60,
        "due_date": "Not finalized",
        "details": "Blocked because the attendance API specification is not finalized."
    },
    {
        "name": "First Release",
        "status": "Planned",
        "progress": 30,
        "due_date": "12-week project timeline",
        "details": "Complete the planned college management modules."
    },
    {
        "name": "Attendance API Specification",
        "status": "Pending",
        "progress": 20,
        "due_date": "This week",
        "details": "Confirm the final attendance API specification."
    },
    {
        "name": "Payment Integration",
        "status": "In Progress",
        "progress": 50,
        "due_date": "Not finalized",
        "details": "Complete and test the fee payment integration."
    },
    {
        "name": "Attendance Backend Integration",
        "status": "Blocked",
        "progress": 40,
        "due_date": "After API confirmation",
        "details": "Connect the attendance UI with backend services."
    },
    {
        "name": "Integration Testing",
        "status": "Pending",
        "progress": 10,
        "due_date": "After integrations stabilize",
        "details": "Prepare and execute integration test cases."
    }
]

for index, milestone in enumerate(milestones):

    with st.container(border=True):

        st.markdown(f"#### {milestone['name']}")

        st.write(f"**Status:** {milestone['status']}")
        st.write(f"**Due Date:** {milestone['due_date']}")

        st.progress(milestone["progress"] / 100)

        st.caption(f"Progress: {milestone['progress']}%")

        st.info(milestone["details"])

       

st.markdown("""
<div class="module-card" style="--accent:#ff9d5c;--progress:65%;">
  <div class="module-icon">⚠️</div>
  <div class="module-title">Risks & Delivery Forecast</div>
  <span class="module-badge">Risk Intelligence</span>
  <div class="module-description">Identify risks, severity, potential impact, dependencies, and delivery challenges from the project documents.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Severity • Impact • Dependencies • Delivery</div>
</div>
""", unsafe_allow_html=True)


if st.button("Analyze Project Risks"):

    risk_chunks = retrieve_documents(
        "project risks blockers delays dependencies "
        "schedule delivery challenges issues problems",
        top_k=6
    )

    if risk_chunks:

        with st.spinner("Analyzing project risks..."):

            risk_analysis = analyze_risks(
                risk_chunks
            )
        st.session_state.risk_analysis = risk_analysis

        st.success(
            "Project risk analysis completed."
        )
        st.markdown("### Risk Analysis")

        with st.expander(
            "View Complete Risk Analysis",
            expanded=True
        ):
            st.write(st.session_state.risk_analysis)
       

    else:

        st.warning(
            "No relevant project risk information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("""
<div class="module-card" style="--accent:#ff6f91;--progress:55%;">
  <div class="module-icon">🚧</div>
  <div class="module-title">Blockers & Action Items</div>
  <span class="module-badge">Blocker Detection</span>
  <div class="module-description">Identify current blockers, their impact, required actions, priorities, and suggested owners.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Issues • Priorities • Owners • Actions</div>
</div>
""", unsafe_allow_html=True)


if st.button("Analyze Blockers & Actions"):

    blocker_chunks = retrieve_documents(
        "current blockers issues pending tasks "
        "action items dependencies problems "
        "delays owners priorities",
        top_k=6
    )

    if blocker_chunks:

        with st.spinner("Analyzing project blockers..."):

            blocker_analysis = analyze_blockers(
                blocker_chunks
            )
            st.session_state.blocker_analysis = blocker_analysis

        st.success(
            "Blocker analysis completed."
        )

        
    else:

        st.warning(
            "No relevant blocker information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("""
<div class="module-card" style="--accent:#b58cff;--progress:80%;">
  <div class="module-icon">📄</div>
  <div class="module-title">Project Documentation</div>
  <span class="module-badge">AI Documentation</span>
  <div class="module-description">Generate a structured project documentation summary from the uploaded project documents.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Overview • Requirements • Status • Summary</div>
</div>
""", unsafe_allow_html=True)


if st.button("Generate Project Documentation"):

    documentation_chunks = retrieve_documents(
        "project overview objectives functional requirements "
        "modules deliverables status risks blockers "
        "pending tasks milestones",
        top_k=10
    )

    if documentation_chunks:

        with st.spinner(
            "Generating project documentation..."
        ):

            documentation = generate_project_documentation(
                documentation_chunks
            )

        st.success(
            "Project documentation generated successfully."
        )

        st.markdown("### Project Documentation")

        st.write(documentation)

    else:

        st.warning(
            "No relevant project information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("""
<div class="module-card" style="--accent:#7ed6a5;--progress:88%;">
  <div class="module-icon">💚</div>
  <div class="module-title">Project Health</div>
  <span class="module-badge">Health Monitoring</span>
  <div class="module-description">Evaluate overall project health based on progress, risks, blockers, and delivery confidence.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Progress • Risks • Blockers • Confidence</div>
</div>
""", unsafe_allow_html=True)


if st.button("Analyze Project Health"):

    health_chunks = retrieve_documents(
        "project status progress risks blockers "
        "delivery schedule completion dependencies",
        top_k=8
    )

    if health_chunks:

        with st.spinner("Analyzing project health..."):

            health_analysis = analyze_project_health(
                health_chunks
            )
            st.session_state.health_analysis = health_analysis

        st.success(
            "Project health analysis completed."
        )

        st.markdown("### Project Health Analysis")

        st.write(health_analysis)

    else:

        st.warning(
            "No relevant project information found. "
            "Please process your project documents first."
        )


# ---------------------------------------------------------
# PROJECT INSIGHTS AND RISK SUMMARY DASHBOARD
# ---------------------------------------------------------

import re

st.markdown("---")

st.markdown("""
<div class="module-card" style="--accent:#5dade2;--progress:95%;">
    <div class="module-icon">📊</div>
    <div class="module-title">
        Project Insights and Risk Summary Dashboard
    </div>
    <span class="module-badge">Project Analytics</span>
    <div class="module-description">
        View project risks, blockers, health status, and AI-generated
        insights in one dashboard.
    </div>
    <div class="module-line"><span></span></div>
    <div class="module-caption">
        KPIs • Risks • Blockers • Health • Insights
    </div>
</div>
""", unsafe_allow_html=True)


def count_priority_items(text, priority):
    if not text:
        return 0

    return len(
        re.findall(
            rf"\b{priority}\b",
            str(text),
            flags=re.IGNORECASE
        )
    )


def count_items(text, keyword):
    if not text:
        return 0

    return len(
        re.findall(
            rf"\b{keyword}\b",
            str(text),
            flags=re.IGNORECASE
        )
    )


dashboard_risk = st.session_state.get(
    "risk_analysis",
    ""
)

dashboard_blocker = st.session_state.get(
    "blocker_analysis",
    ""
)

dashboard_health = st.session_state.get(
    "health_analysis",
    ""
)


if dashboard_risk or dashboard_blocker or dashboard_health:

    total_risk_mentions = count_items(
        dashboard_risk,
        "Risk"
    )

    high_risk_count = count_priority_items(
        dashboard_risk,
        "High"
    )

    total_blocker_mentions = count_items(
        dashboard_blocker,
        "Blocker"
    )

    high_blocker_count = count_priority_items(
        dashboard_blocker,
        "High"
    )

    # ---------------- KPI CARDS ----------------

    st.markdown("### Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Risk Mentions",
            total_risk_mentions
        )

    with col2:
        st.metric(
            "High-Priority Risks",
            high_risk_count
        )

    with col3:
        st.metric(
            "Blocker Mentions",
            total_blocker_mentions
        )

    with col4:
        st.metric(
            "High-Priority Blockers",
            high_blocker_count
        )

    st.markdown("---")

    # ---------------- RISK SUMMARY ----------------

    st.markdown("### Risk Summary")

    risk_summary = {
        "Priority": ["High", "Medium", "Low"],
        "Count": [
            count_priority_items(dashboard_risk, "High"),
            count_priority_items(dashboard_risk, "Medium"),
            count_priority_items(dashboard_risk, "Low")
        ]
    }

    st.bar_chart(
        risk_summary,
        x="Priority",
        y="Count"
    )

    # ---------------- BLOCKER SUMMARY ----------------

    st.markdown("### Blocker Summary")

    blocker_summary = {
        "Priority": ["High", "Medium", "Low"],
        "Count": [
            count_priority_items(dashboard_blocker, "High"),
            count_priority_items(dashboard_blocker, "Medium"),
            count_priority_items(dashboard_blocker, "Low")
        ]
    }

    st.bar_chart(
        blocker_summary,
        x="Priority",
        y="Count"
    )

    # ---------------- PROJECT HEALTH ----------------

    st.markdown("### Project Health")

    if dashboard_health:
        st.info(dashboard_health)
    else:
        st.warning(
            "Project health analysis is not available yet."
        )

    # ---------------- KEY INSIGHTS ----------------

    st.markdown("### Key Project Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        with st.expander("View Risk Insights", expanded=True):
            if dashboard_risk:
                st.write(dashboard_risk)
            else:
                st.info("Run Risk Analysis first.")

    with insight_col2:
        with st.expander(
            "View Blockers and Actions",
            expanded=True
        ):
            if dashboard_blocker:
                st.write(dashboard_blocker)
            else:
                st.info("Run Blocker Analysis first.")

else:

    st.info(
        "Run Risk Analysis, Blocker Analysis, or Project Health "
        "to generate the dashboard."
    )

# ---------------------------------------------------------
# PROJECT INTELLIGENCE
# ---------------------------------------------------------

st.markdown("""
<div class="module-card" style="--accent:#b58cff;--progress:90%;">
  <div class="module-icon">🧠</div>
  <div class="module-title">Project Intelligence</div>
  <span class="module-badge">AI Insights</span>
  <div class="module-description">Review the latest scope, risk, blocker, and project insights generated from your documents.</div>
  <div class="module-line"><span></span></div>
  <div class="module-caption">Scope • Risks • Blockers • Insights</div>
</div>
""", unsafe_allow_html=True)


scope_analysis = st.session_state.get(
    "scope_analysis"
)

risk_analysis = st.session_state.get(
    "risk_analysis"
)

blocker_analysis = st.session_state.get(
    "blocker_analysis"
)

has_intelligence = any([
    scope_analysis,
    risk_analysis,
    blocker_analysis
])

if has_intelligence:

    st.markdown(
        """
        <div class="info-box">
            <b>Project Intelligence Available</b><br><br>
            AI analysis has been generated from the processed
            project documents.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="info-box">
            <b>No project intelligence available yet.</b><br><br>
            Upload and process project documents, then run the
            Scope, Risk, or Blocker analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

