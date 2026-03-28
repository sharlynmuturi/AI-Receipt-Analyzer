# AI Receipt Analyzer

A **Streamlit-based web application** that extracts, processes, and analyzes receipt data using **OCR, deep learning, embeddings, and retrieval-augmented generation (RAG)**. The app allows users to **upload receipts**, **view extracted fields** and **query their data**, getting answers from a combination of structured database and LLM reasoning.


## Features


*   **Receipt Upload & OCR**  
    Upload images (`.jpg`/`.png`) of receipts, with **text extraction** using **Tesseract OCR**.
*   **Deep Learning Field Extraction**  
    Uses **LayoutLMv3** for **token classification**, extracting:
    *   Company
    *   Date
    *   Total
    *   Address
*   **Confidence Scores**  
    Provides confidence values per extracted field for verification.
*   **Database Storage**  
    Stores receipts and extracted metadata in **SQLite** for persistence.
*   **Semantic Search & FAISS Indexing**  
    Embeds receipts as vectors using **SentenceTransformers (`all-MiniLM-L6-v2`)**.
    *   FAISS index for **fast similarity search**
    *   Retrieve most relevant receipts for queries
*   **RAG with LLM**  
    Queries are answered using **Groq API** with **LLM reasoning**, combining structured DB data with embeddings for context-aware responses.
*   **SQL Shortcut Queries**  
    Common questions like:
    *   “Which company did I spend the most at?”
    *   “Total spent?”  
        Can be answered without invoking LLM for speed.


## Technologies Used


| Category | Library/Tool | Purpose |
| --- | --- | --- |
| OCR | pytesseract | Extract text and bounding boxes from receipt images |
| Deep Learning | transformers, torch | LayoutLMv3 for token-level field extraction |
| Embeddings & Search | sentence-transformers, faiss, numpy | Embed receipts and perform similarity search for RAG |
| Database | sqlite3, pandas | Store and query receipt data |
| LLM / RAG | Groq API | Generate answers from retrieved receipts |


## Setup Instructions

### 1\. Clone Repository

```bash

git clone https://github.com/sharlynmuturi/AI-Receipt-Analyzer.git  
cd AI-Receipt-Analyzer
```

### 2\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3\. Environment Variables


Create `.env` in the root directory:

```bash
GROQ_API_KEY=your_api_key_here
```

### 4\. Run the pipeline

Place the dataset in the receipts dir
[Receipt Dataset](https://www.kaggle.com/datasets/dattrinh12/sroie-dataset)

```bash
mkdir receipts
```

Prepare datasets and encode 

```bash
python scripts/prepare_dataset.py
```

Train the model

```bash
python scripts/train_model.py
```

Create the DB, extract receipts, save CSV, build embeddings + FAISS

```bash
python scripts/db_and_embeddings.py
```

### 5\. Run the App

```bash
streamlit run app.py
```


## How It Works

### 1\. OCR & Token Extraction

*   `pytesseract` reads receipt images
*   Extracts **words + bounding boxes**
*   Filters out low-confidence detections (`conf < 40`)

### 2\. LayoutLMv3 Token Classification

*   Input: words + bounding boxes
*   Output: field-level token classification
*   Example labels: `B-COMPANY`, `I-COMPANY`, `B-DATE`, `B-TOTAL`, etc.
*   `extract_fields_with_confidence()` aggregates tokens into fields + computes confidence scores

### 3\. Database Storage

*   Fields are saved in SQLite table `receipts`
*   Raw JSON and confidence scores are also stored for traceability

### 4\. Semantic Embeddings + FAISS

*   Each receipt is converted to a **vector embedding**
*   FAISS index enables **fast retrieval** of similar receipts for a given query
*   New receipts are added to the index on upload


## Performance Optimizations

*   `st.cache_data` and `st.cache_resource` used for caching embeddings and DB lookups
*   Clear cache **only when new receipts are added**: ensures **fast repeated queries** without reloading everything
*   FAISS keeps embeddings in memory for **near-instant retrieval**
*   SQL shortcuts prevent unnecessary LLM calls for simple aggregation queries


## Future Improvements

*   **Batch OCR**: support multiple receipts at once
*   **Advanced analytics**: monthly spending trends, category breakdowns
*   **User authentication**: personalized dashboards