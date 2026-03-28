import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import json
import torch
import torch.nn.functional as F
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from sentence_transformers import SentenceTransformer
import os
from groq import Groq
import numpy as np
import faiss
from pathlib import Path
import easyocr


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "artifacts" / "receipt_model"
INDEX_PATH = BASE_DIR / "artifacts" / "faiss.index"
CSV_PATH = BASE_DIR / "artifacts" / "extracted_receipts.csv"

st.set_page_config(page_title="AI Receipt Analyzer", layout="wide")
st.title("AI Receipt Analyzer")

@st.cache_data
def load_receipts_df():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    else:
        # Return empty dataframe with correct columns
        return pd.DataFrame(columns=["id","company","date","total","address","confidence_json","raw_json"])

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

@st.cache_resource
def load_faiss():
    return faiss.read_index("artifacts/faiss.index")

index = load_faiss()

@st.cache_resource
def load_layoutlm():
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
    model.eval()
    return model, processor

model, processor = load_layoutlm()

model.eval()

# Mapping from IDs to labels
id2label = {0: "O", 1: "B-COMPANY", 2: "I-COMPANY", 3: "B-DATE", 4: "I-DATE",
            5: "B-TOTAL", 6: "I-TOTAL", 7: "B-ADDRESS", 8: "I-ADDRESS"}


# OCR + extraction functions
reader = easyocr.Reader(["en"])

def extract_words_boxes(image):
    results = reader.readtext(np.array(image))
    words, boxes = [], []
    for bbox, text, conf in results:
        if conf < 0.4:
            continue
        x0, y0 = bbox[0]
        x1, y1 = bbox[2]
        # scale to LayoutLM expected 0-1000
        w, h = image.size
        box = [int(1000*x0/w), int(1000*y0/h), int(1000*x1/w), int(1000*y1/h)]
        words.append(text)
        boxes.append(box)
    return words, boxes

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


def save_receipts_df(df):
    df.to_csv(CSV_PATH, index=False)


def save_receipt_to_csv(fields, confidences):
    df = load_receipts_df()
    new_id = 1 if df.empty else df["id"].max() + 1
    new_row = {
        "id": new_id,
        "company": fields.get("COMPANY"),
        "date": fields.get("DATE"),
        "total": fields.get("TOTAL"),
        "address": fields.get("ADDRESS"),
        "confidence_json": json.dumps(confidences),
        "raw_json": json.dumps(fields)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_receipts_df(df)

    
# Receipt processing
def process_receipt(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
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
        probs = F.softmax(outputs.logits, dim=-1)
        preds = probs.argmax(-1)[0]

    word_ids = encoding.word_ids(batch_index=0)
    final_tokens, final_labels, final_scores = [], [], []
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            label_id = preds[idx].item()
            score = probs[0, idx, label_id].item()
            final_tokens.append(words[word_id])
            final_labels.append(id2label[label_id])
            final_scores.append(score)
        prev_word_id = word_id

    fields, confidences = extract_fields_with_confidence(final_tokens, final_labels, final_scores)

    save_receipt_to_csv(fields, confidences)
    
    # Add to FAISS index
    new_row = {
        "company": fields.get("COMPANY"),
        "date": fields.get("DATE"),
        "total": fields.get("TOTAL"),
        "address": fields.get("ADDRESS")
    }
    add_to_index(new_row)

    st.cache_data.clear()
    st.cache_resource.clear()
    
    return fields, confidences
    
def row_to_text(row):
    return f"""
    Receipt:
    - Company: {row['company']}
    - Date: {row['date']}
    - Total: {row['total']} KES
    - Address: {row['address']}
    """

def retrieve(query, df, k=5):
    # Convert DB rows to text
    documents = df.apply(row_to_text, axis=1).tolist()
    
    # Embed query
    query_embedding = embed_model.encode([query])
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Get top documents
    results = [documents[i] for i in indices[0] if i < len(documents)]
    
    return results

def add_to_index(new_row):
    text = row_to_text(new_row)
    
    new_embedding = embed_model.encode([text])
    
    index.add(new_embedding)

    # saving updated index
    faiss.write_index(index, "artifacts/faiss.index")

def try_sql_answer(df, query):
    q = query.lower()
    
    if "most" in q and "company" in q:
        result = df.groupby("company")["total"].sum().idxmax()
        return f"You spent the most at {result}"
    
    if "total spent" in q:
        total = df["total"].sum()
        return f"Total spending is {total}"
    
    return None

# LLM / RAG Query
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def ask_llm(query, retrieved_docs):
    # combine docs into context and return string
    context = "\n\n".join(retrieved_docs)
    prompt = f"""
You are a financial assistant that answers questions about receipts.
Use ONLY the context below.

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
- Prefer using the context.
- If partial information exists, provide the best possible answer.
- Be specific (mention company names, totals, dates).
- Only say "Not enough data" if absolutely nothing is relevant.

Answer clearly and concisely.
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

    
# Streamlit app

page = st.sidebar.selectbox("Select Page", ["Upload", "View & Query"])

# Upload page
if page == "Upload":
    uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg","png"])

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

    if "uploaded_file" in st.session_state:
        file_to_show = st.session_state["uploaded_file"]

        # Display image
        image = Image.open(file_to_show)
        st.image(image, caption="Uploaded Receipt", width=400)

        # Process only if not yet processed
        if "fields" not in st.session_state or "confidences" not in st.session_state:
            with st.spinner("Processing receipt..."):
                fields, confidences = process_receipt(file_to_show)
                st.session_state["fields"] = fields
                st.session_state["confidences"] = confidences

        st.subheader("Extracted Fields")
        st.json(st.session_state["fields"])
        st.subheader("Confidence Scores")
        st.dataframe(st.session_state["confidences"])
        st.success("Receipt saved to database!")

# View & query page
elif page == "View & Query":

    with st.expander("Ask a Question About the Receipts"):
        query = st.text_input("Type your question here")

        if st.button("Ask"):
            df = load_receipts_df()

            if df.empty:
                st.info("No receipts found. Upload some first!")
            else:
                with st.spinner("Generating answer..."):
                    sql_answer = try_sql_answer(df, query)

                    if sql_answer:
                        answer = sql_answer
                    else:
                        retrieved_docs = retrieve(query, df)
                        if not retrieved_docs:
                            retrieved_docs = df.apply(row_to_text, axis=1).tolist()
                        answer = ask_llm(query, retrieved_docs)

                st.session_state.answer = answer

        if "answer" in st.session_state:
            st.subheader("Answer:")
            st.markdown(st.session_state.answer)

    st.subheader("All Receipts")
    df = load_receipts_df()
    if df.empty:
        st.info("No receipts found. Upload some first!")
    else:
        st.dataframe(df)

        if st.button("Export CSV"):
            df.to_csv("receipts_export.csv", index=False)
            st.success("CSV exported!")