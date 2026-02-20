import streamlit as st
import pandas as pd
import json
import requests
import pdfplumber
from pathlib import Path
from sqlalchemy import create_engine, text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import plotly.express as px

# -------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")

API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]

LLM_ENDPOINT = (
    "https://makeathonmj-ai.openai.azure.com/"
    "openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)

# -------------------------------------------------------
# LLM CALL
# -------------------------------------------------------
def call_llm(messages):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "temperature": 0, "max_tokens": 1200}
    r = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        raise Exception(r.text)
    return r.json()["choices"][0]["message"]["content"]

# -------------------------------------------------------
# STRICT SQL PROMPT
# -------------------------------------------------------
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

NOW WRITE THE SQL ONLY for this question:
{question}

SQL:
"""

def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    sql = call_llm([{"role": "user", "content": prompt}]).strip()
    sql = sql.replace("```", "").replace("`", "").strip()
    forbidden = ["DROP", "DELETE", "ALTER", "UPDATE", "INSERT"]
    if any(f in sql.upper() for f in forbidden):
        raise Exception("Unsafe SQL detected.")
    return sql

# -------------------------------------------------------
# VECTOR DB (RAG)
# -------------------------------------------------------
@st.cache_resource
def init_vector_db():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://makeathonmj-ai.openai.azure.com",
        openai_api_key=API_KEY,                 # FIXED NAME
        azure_deployment="text-embedding-3-small",
        api_version="2023-05-15"
    )

    return Chroma(
        collection_name="reports",
        embedding_function=embeddings,
        persist_directory="/tmp/chroma_reports"   # FIXED FOR STREAMLIT CLOUD
    )

vector_db = init_vector_db()

def extract_pdf(pdf):
    with pdfplumber.open(pdf) as p:
        return "\n".join(
            page.extract_text() for page in p.pages if page.extract_text()
        )

def embed_report(text, patient):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vector_db.add_texts(chunks, metadatas=[{"patient": patient}] * len(chunks))
    vector_db.persist()

def ask_report(question, patient):
    docs = vector_db.similarity_search(question, k=3, filter={"patient": patient})
    if not docs:
        return "No relevant report data found."

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clinically:"
    return call_llm([{"role": "user", "content": prompt}])

# -------------------------------------------------------
# AUTO CHART ENGINE
# -------------------------------------------------------
def auto_chart(df):
    if df.shape == (1, 1):
        st.metric(df.columns[0], df.iloc[0, 0])
        return

    numeric = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    cols = df.columns

    chart_type = st.selectbox(
        "Chart Type:", ["Auto", "Bar", "Line", "Pie", "Scatter"], key=str(cols)
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

        st.info("No suitable chart.")
        return

    if chart_type == "Bar":
        st.plotly_chart(px.bar(df, x=cols[0], y=df.columns[-1]), use_container_width=True)

    elif chart_type == "Line":
        st.plotly_chart(px.line(df, x=cols[0], y=df.columns[-1]), use_container_width=True)

    elif chart_type == "Pie" and len(text_cols) >= 1:
        st.plotly_chart(px.pie(df, names=text_cols[0], values=df.columns[-1]), use_container_width=True)

    elif chart_type == "Scatter" and len(numeric) >= 2:
        st.plotly_chart(px.scatter(df, x=numeric[0], y=numeric[1]), use_container_width=True)

# -------------------------------------------------------
# SIDEBAR: DB CONNECTION
# -------------------------------------------------------
st.sidebar.header("🗄 Database Setup")

if "engine" not in st.session_state:
    st.session_state.engine = None

db_name = st.sidebar.text_input("SQLite DB Name:", "demo_healthcare.db")

if st.sidebar.button("Connect"):
    try:
        engine = create_engine(f"sqlite:///{db_name}")
        pd.read_sql("SELECT name FROM sqlite_master", engine)
        st.session_state.engine = engine
        st.sidebar.success("Connected!")
    except Exception as e:
        st.sidebar.error(str(e))

engine = st.session_state.engine

# -------------------------------------------------------
# NAVIGATION
# -------------------------------------------------------
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🧠 SQL Assistant", "📄 Patient RAG", "📘 DB Viewer"]
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "🏠 Home":
    st.title("🏥 Healthcare AI Assistant")
    st.write("AI-powered SQL generation • Clinical RAG • Data Exploration")

# -------------------------------------------------------
# SQL ASSISTANT
# -------------------------------------------------------
elif page == "🧠 SQL Assistant":
    st.title("🧠 Natural Language → SQL Insights")

    if engine is None:
        st.warning("Connect a database first.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.write("### 🔍 Suggested Questions")
    sample_q = [
        "Show all male patients.",
        "How many female patients?",
        "Which patient has the most medications?",
        "List all visits by visit_date.",
        "Show patient count grouped by age."
    ]

    cols = st.columns(3)
    for i, q in enumerate(sample_q):
        with cols[i % 3]:
            if st.button(q):
                sql = generate_sql(q)
                df = pd.read_sql(text(sql), engine)
                st.session_state.history.append({"q": q, "sql": sql, "df": df})
                st.rerun()

    st.write("---")

    q = st.chat_input("Ask your own question...")
    if q:
        sql = generate_sql(q)
        df = pd.read_sql(text(sql), engine)
        st.session_state.history.append({"q": q, "sql": sql, "df": df})
        st.rerun()

    for item in st.session_state.history:
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

# -------------------------------------------------------
# PATIENT RAG
# -------------------------------------------------------
elif page == "📄 Patient RAG":
    st.title("📄 Patient Report Intelligence (RAG)")

    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    patient = st.text_input("Patient Name")

    if pdf and patient:
        text = extract_pdf(pdf)
        embed_report(text, patient)
        st.success("Report embedded!")

    q = st.text_input("Ask about this patient report:")
    if q and patient:
        ans = ask_report(q, patient)
        st.write("### 🧠 Clinical Insight")
        st.write(ans)

# -------------------------------------------------------
# DB VIEWER
# -------------------------------------------------------
elif page == "📘 DB Viewer":
    st.title("📘 Database Viewer")

    if engine is None:
        st.warning("Connect a database first.")
        st.stop()

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)
    tables = tables["name"].tolist()

    for t in tables:
        st.subheader(f"📌 {t}")
        df = pd.read_sql(f"SELECT * FROM {t}", engine)
        st.dataframe(df, use_container_width=True)
