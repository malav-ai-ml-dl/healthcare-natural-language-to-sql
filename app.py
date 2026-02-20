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

API_ENDPOINT = (
    "https://makeathonmj-ai.openai.azure.com/"
    "openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)
API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]


# -------------------------------------------------------
# LLM CALL
# -------------------------------------------------------
def call_llm(messages):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {"messages": messages, "temperature": 0, "max_tokens": 1500}

    res = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(payload))
    if res.status_code != 200:
        raise Exception(res.text)

    return res.json()["choices"][0]["message"]["content"]


# -------------------------------------------------------
# SQL PROMPT
# -------------------------------------------------------
SQL_PROMPT = """
You are an expert SQL generator for a **SQLite healthcare database**.

Your job: Convert ANY natural-language question into **correct SQL**.

STRICT RULES:
1. Output ONLY SQL — no text, no explanation, no markdown, no comments.
2. Use ONLY the tables and columns in the schema.
3. Never invent new tables, columns, or relationships.
4. Always use valid SQLite syntax.
5. If the user asks:
   - “list”, “show”, “display”, “get” → return **full rows**
   - “how many”, “count”, “number of” → return **aggregations**
6. For gender queries:
   - Use gender = 'M' or gender = 'F'
7. For joins, ALWAYS join using patient_id = patients.id
8. For aggregation with details, use GROUP BY.
9. For date-based queries, use visits.visit_date.

SCHEMA:
patients(id, name, age, gender)
visits(id, patient_id, visit_date, reason)
medications(id, patient_id, medication)

EXAMPLES YOU MUST FOLLOW:

Q: How many patients are there?
SQL:
SELECT COUNT(*) AS total_patients FROM patients;

Q: Show all male patients.
SQL:
SELECT * FROM patients WHERE gender = 'M';

Q: How many male and female patients?
SQL:
SELECT gender, COUNT(*) AS patient_count FROM patients GROUP BY gender;

Q: Which patient has the most medications?
SQL:
SELECT p.name, COUNT(m.medication) AS medication_count
FROM patients p
JOIN medications m ON p.id = m.patient_id
GROUP BY p.name
ORDER BY medication_count DESC
LIMIT 1;

Q: List all visits by date.
SQL:
SELECT * FROM visits ORDER BY visit_date;

Q: Show patients with their visit reasons.
SQL:
SELECT p.name, v.reason, v.visit_date
FROM patients p
JOIN visits v ON p.id = v.patient_id
ORDER BY v.visit_date;

USER QUESTION:
{question}

SQL:
"""

def generate_sql(question):
    raw = call_llm([{"role": "user", "content": SQL_PROMPT.format(question=question)}])
    return raw.replace("```", "").strip()


# -------------------------------------------------------
# RAG VECTOR DB
# -------------------------------------------------------
@st.cache_resource
def init_vector_db():
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint="https://makeathonmj-ai.openai.azure.com",
        api_key=API_KEY,
        azure_deployment="text-embedding-3-small",
    )

    # Streamlit Cloud safe directory
    persist_dir = "./.streamlit/chroma_reports"

    return Chroma(
        collection_name="reports",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def extract_pdf(pdf):
    with pdfplumber.open(pdf) as p:
        return "\n".join([pg.extract_text() for pg in p.pages if pg.extract_text()])


def embed_report(text, patient):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = [c for c in splitter.split_text(text) if c.strip()]

    if not chunks:
        raise Exception("PDF extraction returned empty text.")

    vector_db.add_texts(
        texts=chunks,
        metadatas=[{"patient": patient}] * len(chunks)
    )
    vector_db.persist()

def answer_report(q, patient):
    docs = vector_db.similarity_search(
        q,
        k=3,
        filter={"patient": patient}
    )

    if not docs:
        return "No report data found for this patient."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a medical summarization expert.

    Use ONLY the context from patient reports below.  
    If the answer is not in the report, say: 
    "This information is not available in the uploaded report."

    Context:
    {context}

    Question: {q}

    Provide a clear clinical answer:
    """

    return call_llm([{"role": "user", "content": prompt}])

# -------------------------------------------------------
# SMART CHART ENGINE
# -------------------------------------------------------
def auto_chart(df):
    numeric = df.select_dtypes(include="number").columns.tolist()
    text = df.select_dtypes(include="object").columns.tolist()
    cols = df.columns

    # 1) Single metric
    if df.shape == (1,1):
        st.metric(df.columns[0], df.iloc[0,0])
        return

    chart_type = st.selectbox(
        "Select chart type:",
        ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter"],
        key=str(cols)
    )

    def title(x, y=None):
        return f"{y} by {x}" if y else x

    if chart_type == "Auto":
        # A) date → line
        date_cols = [c for c in cols if "date" in c.lower()]
        if date_cols and numeric:
            fig = px.line(df, x=date_cols[0], y=numeric)
            st.plotly_chart(fig, use_container_width=True)
            return

        # B) category + numeric → bar
        if len(text) == 1 and len(numeric) == 1:
            fig = px.bar(df, x=text[0], y=numeric[0])
            st.plotly_chart(fig, use_container_width=True)
            return

        # C) numeric discrete (like age) → bar
        if len(numeric) == 2 and df[numeric[0]].nunique() <= 50:
            fig = px.bar(df, x=numeric[0], y=numeric[1])
            st.plotly_chart(fig, use_container_width=True)
            return

        # D) multi numeric → line
        if len(numeric) >= 2:
            fig = px.line(df[numeric])
            st.plotly_chart(fig, use_container_width=True)
            return

        st.info("No suitable visualization.")
        return

    # MANUAL MODES
    if chart_type == "Bar Chart" and len(numeric) >= 1:
        fig = px.bar(df, x=cols[0], y=numeric[-1])
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Line Chart" and len(numeric) >= 1:
        fig = px.line(df, x=cols[0], y=numeric)
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Pie Chart" and len(text) >= 1 and len(numeric) >= 1:
        fig = px.pie(df, names=text[0], values=numeric[-1])
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Scatter" and len(numeric) >= 2:
        fig = px.scatter(df, x=numeric[0], y=numeric[1])
        st.plotly_chart(fig, use_container_width=True)
        return


# -------------------------------------------------------
# DATABASE SETUP (SQLite only)
# -------------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = None

st.sidebar.title("🗄 Database Setup")

db_name = st.sidebar.text_input("Enter SQLite DB name:", "demo_healthcare.db")

if st.sidebar.button("Connect Database"):
    try:
        engine = create_engine(f"sqlite:///{db_name}")
        # test query
        pd.read_sql("SELECT name FROM sqlite_master", engine)
        st.session_state.engine = engine
        st.sidebar.success(f"Connected to {db_name}")
    except Exception as e:
        st.sidebar.error(str(e))

engine = st.session_state.engine


# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🧠 SQL Assistant", "📄 Patient RAG", "📘 Database Viewer"])


# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <h1 style='text-align:center'>🏥 Healthcare AI Assistant</h1>
    <p style='text-align:center;font-size:18px'>
        Natural Language SQL + Clinical RAG + Database Viewer  
    </p>
    """, unsafe_allow_html=True)

    st.info("👉 Set up your database in the sidebar first!")


# -------------------------------------------------------
# SQL ASSISTANT
# -------------------------------------------------------
elif page == "🧠 SQL Assistant":
    st.title("🧠 Natural Language → SQL Assistant")

    if engine is None:
        st.warning("⚠ Connect a SQLite DB first from the sidebar.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    st.write("### 🔍 Example Questions")
    examples = [
        "How many patients are there?",
        "Show all male patients.",
        "How many male and female patients?",
        "Which patient has the most medications?",
        "Show patient count grouped by age.",
        "List all visits by date."
    ]

    cols = st.columns(3)
    for i, q in enumerate(examples):
        with cols[i % 3]:
            if st.button(q):
                sql = generate_sql(q)
                df = pd.read_sql(text(sql), engine)
                st.session_state.history.append({"q": q, "sql": sql, "df": df})
                st.rerun()

    st.markdown("---")

    q = st.chat_input("Ask something about your data...")
    if q:
        sql = generate_sql(q)
        df = pd.read_sql(text(sql), engine)
        st.session_state.history.append({"q": q, "sql": sql, "df": df})
        st.rerun()

    for item in st.session_state.history:
        with st.container(border=True):
            st.subheader(f"❓ {item['q']}")
            df = item["df"]

            res, chart, sqltab = st.tabs(["📊 Result", "📈 Chart", "🧾 SQL"])

            with res:
                st.dataframe(df, use_container_width=True)

            with chart:
                auto_chart(df)

            with sqltab:
                st.code(item["sql"], language="sql")


# -------------------------------------------------------
# PATIENT RAG
# -------------------------------------------------------
elif page == "📄 Patient RAG":
    st.title("📄 Patient Report Intelligence (RAG)")

    col1, col2 = st.columns(2)
    with col1:
        pdf = st.file_uploader("Upload patient PDF", type=["pdf"])
    with col2:
        patient = st.text_input("Patient Name")

    if pdf and patient:
        text = extract_pdf(pdf)
        embed_report(text, patient)
        st.success("Report embedded!")

    q = st.text_input("Ask a question about this patient report:")
    if q and patient:
        with st.spinner("Analyzing..."):
            ans = answer_report(q, patient)
        st.write("### 🧠 Clinical Insight")
        st.write(ans)


# -------------------------------------------------------
# DATABASE VIEWER
# -------------------------------------------------------
elif page == "📘 Database Viewer":
    st.title("📘 Database Viewer")

    if engine is None:
        st.warning("⚠ Connect a SQLite DB first from the sidebar.")
        st.stop()

    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", engine)["name"].tolist()

    if not tables:
        st.info("No tables found in this database.")
        st.stop()

    for t in tables:
        st.subheader(f"📌 {t}")
        df = pd.read_sql(f"SELECT * FROM {t}", engine)
        st.dataframe(df, use_container_width=True)
