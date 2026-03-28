from pathlib import Path
import sqlite3
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import os
from groq import Groq
from dotenv import load_dotenv
from utils import extract_words_boxes

BASE_DIR = Path(__file__).parent
EVAL_PATH = BASE_DIR / "receipts" / "eval"
EVAL_IMG_PATH = EVAL_PATH / "img"
EVAL_LABEL_PATH = EVAL_PATH / "entities"

MODEL_PATH = BASE_DIR / "artifacts" / "receipt_model"
EXTRACTED_CSV = BASE_DIR / "artifacts" / "extracted_receipts.csv"
EMBEDDINGS_PATH = BASE_DIR / "artifacts" / "doc_embeddings.npy"
INDEX_PATH = BASE_DIR / "artifacts" / "faiss.index"

model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
model.eval()

# Database setup
conn = sqlite3.connect("receipts.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS receipts")
cursor.execute("""
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT,
    date TEXT,
    total REAL,
    address TEXT,
    confidence_json TEXT,
    raw_json TEXT
)
""")
conn.commit()


# Cleaning functions
def clean_total(value):
    if not value: return None
    value = re.sub(r"[^\d.]", "", value)
    try: return float(value)
    except: return None

def clean_date(value):
    if not value: return None
    match = re.search(r"\d{2}/\d{2}/\d{4}", value)
    return match.group(0) if match else value

def clean_address(value):
    if not value: return None
    value = re.sub(r"Document.*", "", value, flags=re.IGNORECASE)
    return value.strip()

def clean_company(value):
    if not value: return None
    return value.strip()


# extraction

def extract_fields_with_confidence(tokens, labels, scores):
    results, confidences = {}, {}
    current_field, buffer, buffer_scores = None, [], []
    for token, label, score in zip(tokens, labels, scores):
        if label.startswith("B-"):
            if current_field:
                results[current_field] = " ".join(buffer)
                confidences[current_field] = sum(buffer_scores)/len(buffer_scores)
            current_field, buffer, buffer_scores = label[2:], [token], [score]
        elif label.startswith("I-") and current_field:
            buffer.append(token)
            buffer_scores.append(score)
        else:
            if current_field:
                results[current_field] = " ".join(buffer)
                confidences[current_field] = sum(buffer_scores)/len(buffer_scores)
                current_field, buffer, buffer_scores = None, [], []
    if current_field:
        results[current_field] = " ".join(buffer)
        confidences[current_field] = sum(buffer_scores)/len(buffer_scores)
    return results, confidences

def save_to_db(fields, confidences):
    cursor.execute("""
    INSERT INTO receipts (company, date, total, address, confidence_json, raw_json)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        clean_company(fields.get("COMPANY")),
        clean_date(fields.get("DATE")),
        clean_total(fields.get("TOTAL")),
        clean_address(fields.get("ADDRESS")),
        json.dumps(confidences),
        json.dumps(fields)
    ))
    conn.commit()


# Inference on eval images
label_list = ["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE","B-TOTAL","I-TOTAL","B-ADDRESS","I-ADDRESS"]
id2label = {i:l for i,l in enumerate(label_list)}

def process_receipt(img_path):
    image = Image.open(img_path).convert("RGB")
    words, boxes = extract_words_boxes(image)

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = probs.argmax(-1)[0]

    word_ids = encoding.word_ids(batch_index=0)
    final_tokens, final_labels, final_scores = [], [], []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id: continue
        label_id = preds[idx].item()
        final_tokens.append(words[word_id])
        final_labels.append(id2label[label_id])
        final_scores.append(probs[0, idx, label_id].item())
        prev_word_id = word_id

    fields, confidences = extract_fields_with_confidence(final_tokens, final_labels, final_scores)
    save_to_db(fields, confidences)
    return fields

# Run extraction
images = list(EVAL_IMG_PATH.glob("*.jpg"))
for img_path in images:
    process_receipt(img_path)

# Export CSV
df = pd.read_sql("SELECT * FROM receipts", conn)
df.to_csv(EXTRACTED_CSV, index=False)

# Build embeddings + FAISS
documents = df.apply(lambda row: f"""
Receipt:
- Company: {row['company']}
- Date: {row['date']}
- Total: {row['total']} KES
- Address: {row['address']}
""", axis=1).tolist()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents, show_progress_bar=True)
np.save(EMBEDDINGS_PATH, doc_embeddings)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings, dtype=np.float32))
faiss.write_index(index, str(INDEX_PATH))


# Loading Groq client
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(query, k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def build_prompt(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    return f"""
You are a financial assistant that answers questions about receipts.
Use ONLY the context below. Do NOT invent information.

Context (each receipt):
{context}

Fields you can use:
- company
- date
- total
- address

Question:
{query}

Instructions:
- Only use the receipts in the context.

Answer clearly and concisely.
"""

def ask_llm(query):
    docs = retrieve(query, k=3)
    prompt = build_prompt(query, docs)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content