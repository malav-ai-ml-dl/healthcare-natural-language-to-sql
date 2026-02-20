import streamlit as st
import pandas as pd
import json
import requests
import pdfplumber
from sqlalchemy import create_engine, text

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import plotly.express as px


# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")

API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
BASE_URL = "https://makeathonmj-ai.openai.azure.com"

LLM_ENDPOINT = (
    f"{BASE_URL}/openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)


# -------------------------------------------------------
# LLM CALLER
# -------------------------------------------------------
def call_llm(messages, max_tokens=1200):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "temperature": 0, "max_tokens": max_tokens}

    r = requests.post(LLM_ENDPOINT, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        raise Exception(r.text)

    return r.json()["choices"][0]["message"]["content"]


# -------------------------------------------------------
# STRICT SQL GENERATOR
# -------------------------------------------------------
SQL_PROMPT = """
You are an expert SQL generator for a STRICT SQLite healthcare database.

RULES:
- Return ONLY executable SQL.
- No markdown, no explanation, no commentary.
- Do NOT hallucinate tables or columns.
- Gender is ONLY 'M' or 'F'.
- Use correct table/column names.

DATABASE SCHEMA:
patients(id, name, age, gender)
visits(id, patient_id, visit_date, reason)
medications(id, patient_id, medication)

EXAMPLES:
Q: Show all male patients.
SELECT * FROM patients WHERE gender='M';

Q: How many patients?
SELECT COUNT(*) AS count FROM patients;

Q: How many male and female patients?
SELECT gender, COUNT(*) AS count FROM patients GROUP BY gender;

Q: Patients with most medications.
SELECT p.name, COUNT(m.medication) AS count
FROM patients p
JOIN medications m ON p.id = m.patient_id
GROUP BY p.name
ORDER BY count DESC;

NOW generate SQL ONLY for this question:
{question}

SQL:
"""


def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    sql = call_llm([{"role": "user", "content": prompt}]).strip()

    # Cleanup
    sql = sql.replace("```", "").replace("`", "").strip()

    # Prevent unsafe SQL
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    if any(x in sql.upper() for x in forbidden):
        raise Exception("Unsafe SQL detected.")

    return sql


# -------------------------------------------------------
# RAG – VECTOR DB INITIALIZATION
# -------------------------------------------------------
@st.cache_resource
def init_vector_db():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=BASE_URL,
        api_key=API_KEY,
        azure_deployment="text-embedding-3-small"
    )

    return Chroma(
        collection_name="reports",
        embedding_function=embeddings,
        persist_directory="chroma_reports"
    )


vector_db = init_vector_db()


# -------------------------------------------------------
# PDF EXTRACT
# -------------------------------------------------------
def extract_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(
            pg.extract_text() for pg in pdf.pages if pg.extract_text()
        )


def embed_report(text, patient_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vector_db.add_texts(
        texts=chunks,
        metadatas=[{"patient": patient_name}] * len(chunks)
    )
    vector_db.persist()


def ask_report(question, patient_name):
    docs = vector_db.similarity_search(
        question, k=3, filter={"patient": patient_name}
    )

    if not docs:
        return "No relevant clinical information found."

    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clinically:"

    return call_llm([{"role": "user", "content": prompt}])


# -------------------------------------------------------
# AUTO CHARTING
# -------------------------------------------------------
def auto_chart(df):
    if df.shape == (1, 1):
        st.metric(df.columns[0], df.iloc[0, 0])
        return

    numeric = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()
    cols = df.columns.tolist()

    chart = st.selectbox(
        "Chart Type:",
        ["Auto", "Bar", "Line", "Pie", "Scatter"],
        key=str(cols),
    )

    if chart == "Auto":
        if any("date" in c.lower() for c in cols) and numeric:
            st.plotly_chart(
                px.line(df, x=[c for c in cols if "date" in c.lower()][0], y=numeric),
                use_container_width=True,
            )
            return

        if len(text_cols) == 1 and len(numeric) == 1:
            st.plotly_chart(px.bar(df, x=text_cols[0], y=numeric[0]), use_container_width=True)
            return

        if len(numeric) >= 2:
            st.plotly_chart(px.line(df[numeric]), use_container_width=True)
            return

        st.info("No suitable chart.")
        return

    if chart == "Bar":
        st.plotly_chart(px.bar(df, x=cols[0], y=numeric[-1]), use_container_width=True)
    elif chart == "Line":
        st.plotly_chart(px.line(df, x=cols[0], y=numeric[-1]), use_container_width=True)
    elif chart == "Pie" and text_cols:
        st.plotly_chart(px.pie(df, names=text_cols[0], values=numeric[-1]), use_container_width=True)
    elif chart == "Scatter" and len(numeric) >= 2:
        st.plotly_chart(px.scatter(df, x=numeric[0], y=numeric[1]), use_container_width=True)


# -------------------------------------------------------
# SIDEBAR DB CONNECTION
# -------------------------------------------------------
st.sidebar.header("🗄 Database Setup")

if "engine" not in st.session_state:
    st.session_state.engine = None

db_name = st.sidebar.text_input("SQLite DB Name", "demo_healthcare.db")

if st.sidebar.button("Connect"):
    try:
        engine = create_engine(f"sqlite:///{db_name}")
        pd.read_sql("SELECT name FROM sqlite_master", engine)
        st.session_state.engine = engine
        st.sidebar.success("Connected successfully!")
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
# HOME
# -------------------------------------------------------
if page == "🏠 Home":
    st.title("🏥 Healthcare AI Assistant")
    st.write("Natural language SQL • Clinical RAG • Data Visualization")


# -------------------------------------------------------
# SQL ASSISTANT
# -------------------------------------------------------
elif page == "🧠 SQL Assistant":
    st.title("🧠 Natural Language → SQL Engine")

    if engine is None:
        st.warning("Connect a database first.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.write("### 🔍 Suggested Questions")
    suggestions = [
        "Show all male patients.",
        "How many female patients?",
        "Which patient has the most medications?",
        "List all visits by visit_date.",
        "Show patient count grouped by age."
    ]

    cols = st.columns(3)
    for i, q in enumerate(suggestions):
        if cols[i % 3].button(q):
            sql = generate_sql(q)
            df = pd.read_sql(text(sql), engine)
            st.session_state.history.append({"q": q, "sql": sql, "df": df})
            st.rerun()

    st.write("---")

    user_q = st.chat_input("Ask your own question...")
    if user_q:
        sql = generate_sql(user_q)
        df = pd.read_sql(text(sql), engine)
        st.session_state.history.append({"q": user_q, "sql": sql, "df": df})
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
    st.title("📄 Patient Report Intelligence")

    pdf = st.file_uploader("Upload Report (PDF)", type=["pdf"])
    patient = st.text_input("Patient Name")

    if pdf and patient:
        text = extract_pdf(pdf)
        embed_report(text, patient)
        st.success("Report embedded successfully!")

    q = st.text_input("Ask a clinical question:")
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

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'",
        engine
    )["name"].tolist()

    for t in tables:
        st.subheader(f"📌 {t}")
        df = pd.read_sql(f"SELECT * FROM {t}", engine)
        st.dataframe(df, use_container_width=True)
