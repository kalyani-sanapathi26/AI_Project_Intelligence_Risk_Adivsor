import streamlit as st
from src.document_loader import extract_text
from src.chunking import split_documents
from src.embeddings import generate_embeddings
from src.vector_store import store_embeddings
from src.rag_pipeline import retrieve_documents
from src.llm import generate_answer
from src.scope_agent import analyze_scope
from src.risk_agent import analyze_risks
from src.blocker_agent import analyze_blockers
from src.documentation_agent import generate_project_documentation
from src.health_agent import analyze_project_health



st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="📊",
    layout="wide"
)

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
st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">AI Project Advisor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Project Intelligence Platform</div>',
        unsafe_allow_html=True
    )

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Documents",
            "RAG Pipeline",
            "Project Scope",
            "Risks & Forecast",
            "Blockers & Actions",
            "Documentation",
            "Project Health",
            "AI Assistant"
        ],
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
st.markdown(
    '<div class="section-title">Upload Project Documents</div>',
    unsafe_allow_html=True
)

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
st.markdown(
    '<div class="section-title">RAG Pipeline</div>',
    unsafe_allow_html=True
)

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

st.markdown(
    '<div class="section-title">Ask Your Project</div>',
    unsafe_allow_html=True
)

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
            answer = generate_answer(
                query,
                results
            )

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

st.markdown("## Project Scope & Deliverables")

st.write(
    "Analyze the uploaded project documents to identify "
    "the project objective, scope, modules, deliverables, "
    "requirements, and milestones."
)

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

        st.markdown("### Scope Analysis")

        st.write(scope_analysis)

    else:

        st.warning(
            "No relevant project information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("## Risks & Delivery Forecast")

st.write(
    "Analyze the project documents to identify risks, "
    "severity, potential impact, dependencies, and "
    "delivery challenges."
)

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

        st.write(risk_analysis)

    else:

        st.warning(
            "No relevant project risk information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("## Blockers & Action Items")

st.write(
    "Identify current project blockers, their impact, "
    "required actions, priorities, and suggested owners."
)

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

        st.markdown("### Blockers & Recommended Actions")

        st.write(blocker_analysis)

    else:

        st.warning(
            "No relevant blocker information found. "
            "Please process your project documents first."
        )

st.markdown("---")

st.markdown("## Project Documentation")

st.write(
    "Generate a structured project documentation summary "
    "from the uploaded project documents."
)

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

st.markdown("## Project Health")

st.write(
    "Evaluate the overall project health based on "
    "progress, risks, blockers, and delivery confidence."
)

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
# PROJECT INTELLIGENCE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Project Intelligence</div>',
    unsafe_allow_html=True
)

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

    if scope_analysis:

        st.markdown("### Project Scope")

        st.write(scope_analysis)

    if risk_analysis:

        st.markdown("### Project Risks")

        st.write(risk_analysis)

    if blocker_analysis:

        st.markdown("### Blockers & Actions")

        st.write(blocker_analysis)

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