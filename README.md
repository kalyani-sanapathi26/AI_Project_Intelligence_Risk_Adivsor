# AI Project Intelligence & Risk Advisor

An AI-powered project management application that helps teams understand project scope, identify risks and blockers, evaluate project health, and generate structured project documentation using project-related information.

## Overview

The **AI Project Intelligence & Risk Advisor** is designed to make project monitoring and analysis easier by bringing project documents, AI-powered insights, risk analysis, and project reporting together in one Streamlit application.

The system uses uploaded project documents and analysis results to help users understand project status, identify potential issues, and plan the next actions.

## Key Features

- **Document Processing:** Upload and process project documents for further analysis.
- **RAG Pipeline:** Retrieve relevant information from project documents and answer project-related questions using AI.
- **Project Scope Analysis:** Identify project objectives, scope, requirements, deliverables, and milestones.
- **Risk Analysis:** Identify project risks, assess their severity, and provide mitigation guidance.
- **Blocker Analysis:** Identify project blockers and recommended actions to resolve them.
- **Project Health Analysis:** Evaluate project health across multiple dimensions and display an overall health score.
- **AI Project Assistant:** Ask questions about project information and receive context-based answers.
- **Documentation Generation:** Generate structured project documentation containing User Stories, Risk Register, Action Items, and Project Summary.
- **Interactive Dashboard:** View project health, risks, blockers, and other project insights in a visual layout.
- **Report Downloads:** Download supported project documentation and dashboard reports in available formats.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web application |
| Large Language Models (LLMs) | AI-powered analysis and responses |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector storage and retrieval |
| Retrieval-Augmented Generation (RAG) | Context-based question answering |
| Git and GitHub | Version control and project hosting |

The application supports configured LLM providers, including Groq, Google Gemini, and Ollama.

## Project Structure

```text
AI_Project_Intelligence_Risk_Advisor/
├── app.py
├── src/
│   ├── blocker_agent.py
│   ├── chunking.py
│   ├── document_loader.py
│   ├── documentation_agent.py
│   ├── embeddings.py
│   ├── health_agent.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── risk_agent.py
│   ├── scope_agent.py
│   └── vector_store.py
├── data/
├── utils/
├── Agile_Documents/
├── requirements.txt
├── .gitignore
└── README.md
```

*The structure above describes the main application components; your local repository may contain additional files.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kalyani-sanapathi26/AI_Project_Intelligence_Risk_Adivsor.git
```

### 2. Open the project directory

```bash
cd AI_Project_Intelligence_Risk_Adivsor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the AI provider

Configure the API key for your chosen cloud provider using environment variables or a local `.env` file, as supported by the project. Alternatively, configure Ollama if using a local model.

**Security:** Never commit API keys, secrets, or your `.env` file to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## How to Use

1. Open the application.
2. Upload your project documents.
3. Process the documents to prepare them for analysis.
4. Explore the RAG Pipeline and ask questions about the project.
5. Review project scope, risks, blockers, and project health.
6. Generate structured project documentation.
7. Explore the dashboard and download available reports.

## Agile Documentation

The repository also contains Agile project documents, including applicable planning, tracking, meeting, testing, and retrospective records.

These documents support project planning, sprint tracking, defect management, test planning, and continuous improvement.

## Future Enhancements

- More interactive project analytics and visualizations.
- Improved risk prioritization and mitigation recommendations.
- Enhanced project progress tracking and delivery forecasting.
- More comprehensive project reporting.
- Improved AI-generated insights and documentation quality.

## License

Add the license information here once a license has been selected for the repository.

## Author

**Kalyani Sanapathi**

GitHub: [kalyani-sanapathi26](https://github.com/kalyani-sanapathi26)

---

*AI Project Intelligence & Risk Advisor — turning project information into actionable insights.*
