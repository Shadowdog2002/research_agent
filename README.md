# Research Agent — Setup & Run Instructions

## Requirements

- Python 3.9 or higher
- Internet connection (for arXiv searches and the Anthropic API)

---

## Steps to Run

### 0. Unzip the project

The project is submitted as a zip file. Extract it to a folder before continuing.

### 1. Create a virtual environment (recommended)

A virtual environment keeps this project's dependencies isolated from your system Python, preventing version conflicts with other installed packages.

```bash
python -m venv venv
```

Then activate it (this project was developed on Windows):

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run will automatically download two small ML models (~100 MB total) which takes 3–5 minutes. This only happens once — they are cached locally after that.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser automatically. If it doesn't, navigate to `http://localhost:8501`.

> The `.env` file containing the API key is included in the submission — no setup needed.

---

## How to Use

### Login

Enter any username on the login screen. Each username gets its own isolated knowledge base, saved reviews, and chat history. You can use any name — nothing is password-protected.

### Pages

1. **Search Papers** — Enter a research topic and click "Run Pipeline". The agent searches arXiv, selects the 5 most relevant papers using a cross-encoder reranker, summarises each one, generates a literature review, runs a CriticAgent to review the draft, then produces a revised final review. Also generates a research gap analysis with suggested follow-up searches. Takes 3–7 minutes depending on paper length.

2. **Literature Review** — View the final revised literature review alongside a collapsed "CriticAgent Notes" section showing the critique used to improve it. Also shows the research gap analysis and suggested arXiv queries. Previous reviews are saved per user and shown below.

3. **Ask Questions** — Ask natural-language questions about the papers in your knowledge base. Answers are grounded in retrieved paper excerpts and include citations and a confidence level (High / Medium / Low).

4. **Knowledge Base** — View all ingested papers, delete individual papers, upload your own PDFs, or search arXiv to add individual papers. You can also generate a literature review from any selection of up to 6 papers already in the knowledge base.
