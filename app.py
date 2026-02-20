import streamlit as st
import pandas as pd
import json
import requests
import pdfplumber
from pathlib import Path
from sqlalchemy import create_engine, text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import faiss
import numpy as np
import pickle

import plotly.express as px


# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")

API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
LLM_ENDPOINT = (
    "https://makeathonmj-ai.openai.azure.com/"
    "openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)

INDEX_FILE = Path("faiss_index.bin")
METADATA_FILE = Path("metadata.pkl")


# ───────────────────────────────────────────────
# LLM CALL
# ───────────────────────────────────────────────
def call_llm(messages):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "temperature": 0, "max_tokens": 1200}
    r = requests.post(LLM_ENDPOINT, headers=headers, json=payload)
    if r.status_code != 200:
        raise Exception(r.text)
    return r.json()["choices"][0]["message"]["content"]


# ───────────────────────────────────────────────
# STRICT SQL PROMPT
# ───────────────────────────────────────────────
SQL_PROMPT = """
You are a SQL generator for a STRICT SQLite healthcare database.

RULES:
- Return ONLY executable SQL (no explanation, no markdown).
- Do NOT hallucinate tables or columns.
- Gender values are ONLY 'M' or 'F'.
- Use COUNT(*) AS count for counting.
- Use GROUP BY when grouping.
- Use SELECT * for listing rows.
- Use correct table and column names exactly as below.

SCHEMA:
patients(id, name, age, gender)
visits(id, patient_id, visit_date, reason)
medications(id, patient_id, medication)

GOOD EXAMPLES:
Q: How many patients?
A:
SELECT COUNT(*) AS count FROM patients;

Q: Show all male patients.
A:
SELECT * FROM patients WHERE gender='M';

Q: How many male and female patients?
A:
SELECT gender, COUNT(*) AS count FROM patients GROUP BY gender;

Q: Which patient has the most medications?
A:
SELECT p.name, COUNT(m.medication) AS count
FROM patients p
JOIN medications m ON p.id = m.patient_id
GROUP BY p.name
ORDER BY count DESC;

NOW WRITE THE SQL ONLY for this question:
{question}

SQL:
"""


def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    sql = call_llm([{"role": "user", "content": prompt}]).strip()
    sql = sql.replace("```", "").replace("`", "").strip()

    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    if any(f in sql.upper() for f in forbidden):
        raise Exception("Unsafe SQL detected.")

    return sql


# ───────────────────────────────────────────────
# FAISS VECTOR STORE – Improved version
# ───────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint="https://makeathonmj-ai.openai.azure.com",
    api_key=API_KEY,
    azure_deployment="text-embedding-3-small"
)

# We store one FAISS index + metadata per patient to avoid mixing documents
INDEX_DIR = Path("faiss_indices")
INDEX_DIR.mkdir(exist_ok=True)


def get_patient_key(patient: str) -> str:
    """Normalize patient name for file key (lowercase, strip, replace spaces)"""
    if not patient:
        return "unknown"
    return patient.strip().lower().replace(" ", "_").replace(".", "")


@st.cache_resource(show_spinner=False)
def load_patient_index(patient: str):
    key = get_patient_key(patient)
    index_path = INDEX_DIR / f"{key}_index.faiss"
    meta_path = INDEX_DIR / f"{key}_meta.pkl"

    if not index_path.exists() or not meta_path.exists():
        return None, []

    try:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        if len(metadata) == 0:
            return None, []
        return index, metadata
    except Exception as e:
        st.warning(f"Failed to load index for {patient}: {e}")
        return None, []


def save_patient_index(patient: str, index, metadata):
    key = get_patient_key(patient)
    index_path = INDEX_DIR / f"{key}_index.faiss"
    meta_path = INDEX_DIR / f"{key}_meta.pkl"

    try:
        faiss.write_index(index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        st.success(f"Saved index for {patient} ({len(metadata)} chunks)")
    except Exception as e:
        st.error(f"Failed to save index for {patient}: {e}")


def embed_report(text: str, patient: str):
    if not text.strip():
        st.error("No text extracted from PDF.")
        return

    patient = patient.strip()
    if not patient:
        st.error("Please enter a patient name.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not chunks:
        st.warning("Splitter produced no chunks.")
        return

    with st.spinner(f"Creating {len(chunks)} embeddings..."):
        try:
            emb_list = embeddings.embed_documents(chunks)
        except Exception as e:
            st.error(f"Embedding API failed: {str(e)}")
            return

    emb_array = np.array(emb_list).astype("float32")
    if emb_array.shape[0] != len(chunks):
        st.error("Embedding count mismatch!")
        return

    dim = emb_array.shape[1]

    # Load existing or create new
    current_index, current_meta = load_patient_index(patient)

    if current_index is None:
        new_index = faiss.IndexFlatL2(dim)
        new_meta = []
    else:
        new_index = current_index
        new_meta = current_meta

    new_index.add(emb_array)
    new_meta.extend([{"patient": patient, "text": chunk} for chunk in chunks])

    save_patient_index(patient, new_index, new_meta)

    # Debug info
    st.info(f"Embedded **{len(chunks)}** chunks for **{patient}**")
    with st.expander("First 2 chunks preview"):
        for i, chunk in enumerate(chunks[:2], 1):
            st.markdown(f"**Chunk {i}** ({len(chunk)} chars):  \n{chunk[:400]}...")


def ask_report(question: str, patient: str):
    if not patient.strip():
        return "Please enter a patient name."

    patient = patient.strip()
    index, metadata = load_patient_index(patient)

    if index is None or len(metadata) == 0:
        return f"No embedded report found for patient '{patient}'.\n\nTry uploading and embedding a PDF first."

    with st.spinner("Searching..."):
        try:
            q_emb = embeddings.embed_query(question)
            q_array = np.array([q_emb]).astype("float32")

            # Search more candidates → filter later
            distances, indices = index.search(q_array, k=8)

            relevant_chunks = []
            scores = []

            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                chunk_data = metadata[idx]
                # Still double-check patient (in case of index mixup)
                if chunk_data["patient"] == patient:
                    relevant_chunks.append(chunk_data["text"])
                    scores.append(float(dist))

            if not relevant_chunks:
                return f"No relevant chunks found for '{patient}' (even though index exists)."

            # Sort by distance (lower = better)
            sorted_pairs = sorted(zip(relevant_chunks, scores), key=lambda x: x[1])
            context_chunks = [text for text, _ in sorted_pairs[:4]]  # top 4

            context = "\n\n───\n\n".join(context_chunks)

            prompt = f"""You are a helpful clinical assistant.
Use **only** the following context from the patient's report.
If the information is not in the context, say so.

Context:
{context}

Question: {question}

Answer concisely and clinically:"""

            return call_llm([{"role": "user", "content": prompt}])

        except Exception as e:
            return f"Error during retrieval: {str(e)}"


# ───────────────────────────────────────────────
# AUTO CHART
# ───────────────────────────────────────────────
def auto_chart(df):
    if df.empty:
        st.info("No data to chart.")
        return

    if df.shape == (1, 1):
        st.metric(df.columns[0], df.iloc[0, 0])
        return

    numeric = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    cols = df.columns

    chart_type = st.selectbox(
        "Chart Type:", ["Auto", "Bar", "Line", "Pie", "Scatter"], key="chart_" + "_".join(cols)
    )

    if chart_type == "Auto":
        date_cols = [c for c in cols if "date" in c.lower()]
        if date_cols and numeric:
            st.plotly_chart(px.line(df, x=date_cols[0], y=numeric), use_container_width=True)
            return
        if len(text_cols) == 1 and len(numeric) == 1:
            st.plotly_chart(px.bar(df, x=text_cols[0], y=numeric[0]), use_container_width=True)
            return
        if len(numeric) >= 2:
            st.plotly_chart(px.line(df[numeric]), use_container_width=True)
            return
        st.info("No suitable automatic chart found.")
        return

    if chart_type == "Bar" and len(cols) >= 2:
        st.plotly_chart(px.bar(df, x=cols[0], y=cols[-1]), use_container_width=True)
    elif chart_type == "Line" and len(cols) >= 2:
        st.plotly_chart(px.line(df, x=cols[0], y=cols[-1]), use_container_width=True)
    elif chart_type == "Pie" and len(text_cols) >= 1 and len(numeric) >= 1:
        st.plotly_chart(px.pie(df, names=text_cols[0], values=numeric[0]), use_container_width=True)
    elif chart_type == "Scatter" and len(numeric) >= 2:
        st.plotly_chart(px.scatter(df, x=numeric[0], y=numeric[1]), use_container_width=True)


# ───────────────────────────────────────────────
# SIDEBAR DB CONNECTION
# ───────────────────────────────────────────────
st.sidebar.header("🗄 Database Setup")

if "engine" not in st.session_state:
    st.session_state.engine = None

db_name = st.sidebar.text_input("SQLite DB Name:", "demo_healthcare.db")

if st.sidebar.button("Connect"):
    try:
        engine = create_engine(f"sqlite:///{db_name}")
        pd.read_sql("SELECT name FROM sqlite_master", engine)  # test connection
        st.session_state.engine = engine
        st.sidebar.success("Connected!")
    except Exception as e:
        st.sidebar.error(str(e))

engine = st.session_state.engine


# ───────────────────────────────────────────────
# NAVIGATION
# ───────────────────────────────────────────────
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🧠 SQL Assistant", "📄 Patient RAG", "📘 DB Viewer"]
)


# ───────────────────────────────────────────────
# PAGES
# ───────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("""
    AI-powered features:
    - Natural language → SQL queries
    - PDF report intelligence (RAG with FAISS)
    - Simple database exploration
    """)


elif page == "🧠 SQL Assistant":
    st.title("🧠 Natural Language → SQL Insights")

    if engine is None:
        st.warning("Connect a database first in the sidebar.")
        st.stop()

    if "sql_history" not in st.session_state:
        st.session_state.sql_history = []

    st.write("### 🔍 Suggested Questions")
    sample_q = [
        "Show all male patients.",
        "How many female patients?",
        "Which patient has the most medications?",
        "List all visits by visit_date.",
        "Show patient count grouped by age.",
    ]

    cols = st.columns(3)
    for i, q in enumerate(sample_q):
        with cols[i % 3]:
            if st.button(q, key=f"sample_{i}"):
                try:
                    sql = generate_sql(q)
                    df = pd.read_sql(text(sql), engine)
                    st.session_state.sql_history.append({"q": q, "sql": sql, "df": df})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.write("---")

    user_q = st.chat_input("Ask your own question about the database...")
    if user_q:
        try:
            sql = generate_sql(user_q)
            df = pd.read_sql(text(sql), engine)
            st.session_state.sql_history.append({"q": user_q, "sql": sql, "df": df})
            st.rerun()
        except Exception as e:
            st.error(f"Error generating/executing SQL: {e}")

    for item in reversed(st.session_state.sql_history):  # newest first
        with st.container(border=True):
            st.subheader(f"❓ {item['q']}")
            df = item["df"]

            tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Chart", "🧾 SQL"])
            with tab1:
                st.dataframe(df, use_container_width=True)
            with tab2:
                auto_chart(df)
            with tab3:
                st.code(item["sql"], language="sql")


elif page == "📄 Patient RAG":
    st.title("📄 Patient Report Intelligence (RAG)")

    pdf = st.file_uploader("Upload PDF report", type=["pdf"])
    patient = st.text_input("Patient Name (used as filter)", "")

    if pdf and patient and st.button("Embed this report"):
        with st.spinner("Extracting & embedding..."):
            try:
                text = ""
                with pdfplumber.open(pdf) as p:
                    for page in p.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    embed_report(text, patient.strip())
                else:
                    st.error("No readable text found in PDF.")
            except Exception as e:
                st.error(f"Embedding failed: {e}")

    q = st.text_input("Ask a question about this patient's report:")
    if q and patient:
        with st.spinner("Searching reports..."):
            try:
                ans = ask_report(q, patient.strip())
                st.markdown("### 🧠 Clinical Insight")
                st.write(ans)
            except Exception as e:
                st.error(f"Query failed: {e}")


elif page == "📘 DB Viewer":
    st.title("📘 Database Viewer")

    if engine is None:
        st.warning("Connect a database first in the sidebar.")
        st.stop()

    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            engine
        )["name"].tolist()

        for t in tables:
            st.subheader(f"📌 Table: {t}")
            df = pd.read_sql(f"SELECT * FROM {t} LIMIT 1000", engine)
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not list tables: {e}")
