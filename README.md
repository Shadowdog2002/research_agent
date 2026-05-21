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

> The first run will automatically download two small ML models (~100 MB total) which takes 3-5 minutes. This only happens once. They are cached locally after that.

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser automatically. If it doesn't, navigate to `http://localhost:8501`.

> The `.env` file containing the API key is included in the submission so there is no setup needed.

---

## How to Use

1. **Search Papers** — Enter a research topic and click "Run Pipeline". The agent will search arXiv, select the 5 most relevant papers, summarise each one, and generate a literature review. This takes 2–5 minutes depending on paper length.

2. **Literature Review** — View the generated review and research gap analysis. Previous reviews are saved and shown below.

3. **Ask Questions** — Ask natural-language questions about the papers in your knowledge base. Answers include citations and a confidence level.

4. **Knowledge Base** — View all ingested papers, upload your own PDFs, or search arXiv to add individual papers.
