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

import plotly.express as px


# ───────────────────────────────────────────────
# PAGE CONFIG & SECRETS
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Data Assistant – Clinical & Data Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
LLM_ENDPOINT = (
    "https://makeathonmj-ai.openai.azure.com/"
    "openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)


# ───────────────────────────────────────────────
# LLM CALL
# ───────────────────────────────────────────────
def call_llm(messages, temperature=0.1, max_tokens=1800):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"LLM API error: {str(e)}")
        return None


# ───────────────────────────────────────────────
# SQL GENERATION
# ───────────────────────────────────────────────
SQL_PROMPT = """You are a precise SQLite SQL generator for a healthcare database.
Return **ONLY** valid SQL – no explanations, no markdown, no comments.

Rules:
- Use exactly the schema below – do NOT invent tables/columns
- Gender: only 'M' or 'F'
- Use COUNT(*) AS count
- GROUP BY for aggregations
- SELECT * when showing rows
- Safe SQL only – no DDL/DML except SELECT

SCHEMA:
patients(id, name, age, gender)
visits(id, patient_id, visit_date, reason)
medications(id, patient_id, medication)

Examples:
Q: How many patients? →
SELECT COUNT(*) AS count FROM patients;

Q: Male patients → 
SELECT * FROM patients WHERE gender = 'M';

NOW – generate SQL ONLY for:
{question}
"""


def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    response = call_llm([{"role": "user", "content": prompt}], temperature=0.0)
    if not response:
        return None
    sql = response.strip().replace("```sql", "").replace("```", "").strip()
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    if any(word in sql.upper() for word in forbidden):
        raise ValueError("Unsafe SQL pattern detected")
    return sql


# ───────────────────────────────────────────────
# EMBEDDINGS & IN-MEMORY RAG
# ───────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint="https://makeathonmj-ai.openai.azure.com",
    api_key=API_KEY,
    azure_deployment="text-embedding-3-small"
)

if "patient_docs" not in st.session_state:
    st.session_state.patient_docs = {}  # patient → {"index", "metadata", "filename", "loaded_at"}


def process_and_index_report(pdf_file, patient_name: str):
    patient_name = patient_name.strip()
    if not patient_name:
        return False, "Please enter patient name"

    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n\n"

        if not text.strip():
            return False, "No readable text found in PDF"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=140,
            separators=["\n\n", "\n", ". ", "!", "?", " ", ""]
        )
        chunks = splitter.split_text(text.strip())

        if not chunks:
            return False, "Could not split document into chunks"

        emb_list = embeddings.embed_documents(chunks)
        emb_array = np.array(emb_list).astype("float32")

        index = faiss.IndexFlatL2(emb_array.shape[1])
        index.add(emb_array)

        metadata = [{"patient": patient_name, "text": c} for c in chunks]

        st.session_state.patient_docs[patient_name] = {
            "index": index,
            "metadata": metadata,
            "filename": pdf_file.name,
            "loaded_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        }

        return True, f"Report indexed successfully ({len(chunks)} sections)"

    except Exception as e:
        return False, f"Processing failed: {str(e)}"


def query_report(question: str, patient: str):
    patient = patient.strip()
    if not patient or patient not in st.session_state.patient_docs:
        return None, "No report loaded for this patient"

    data = st.session_state.patient_docs[patient]
    index = data["index"]
    metadata = data["metadata"]

    try:
        q_emb = embeddings.embed_query(question)
        q_array = np.array([q_emb]).astype("float32")

        D, I = index.search(q_array, k=10)

        relevant = []
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            relevant.append(metadata[idx]["text"])

        if not relevant:
            return None, "No relevant information found in report"

        context = "\n\n─────────────────────────────\n\n".join(relevant[:6])

        prompt = f"""You are an experienced clinical decision support assistant.

Rules:
- Answer using **only** the provided report excerpts.
- Use precise medical language.
- Structure answer professionally:
  ### Clinical Summary
  ### Key Findings
  ### Interpretation & Implications
  ### Missing / Unclear Information (if relevant)
- Be cautious — do not invent facts.
- If question is unrelated to report, say so politely.

Report content:
{context}

Question:
{question}

Answer:"""

        answer = call_llm([{"role": "user", "content": prompt}], temperature=0.15)
        return answer, None

    except Exception as e:
        return None, f"Query error: {str(e)}"


# ───────────────────────────────────────────────
# AUTO VISUALIZATION
# ───────────────────────────────────────────────
def render_auto_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No data available for visualization.")
        return

    if len(df) == 1 and len(df.columns) == 1:
        st.metric(df.columns[0], df.iloc[0, 0])
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    chart_type = st.selectbox(
        "Visualization type",
        ["Auto", "Bar", "Line", "Pie", "Scatter"],
        key=f"chart_{hash(df.to_string())}"
    )

    if chart_type == "Auto":
        date_like = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if date_like and numeric_cols:
            st.plotly_chart(
                px.line(df, x=date_like[0], y=numeric_cols, height=420),
                use_container_width=True
            )
            return
        if len(cat_cols) == 1 and len(numeric_cols) == 1:
            st.plotly_chart(
                px.bar(df, x=cat_cols[0], y=numeric_cols[0], height=420),
                use_container_width=True
            )
            return
        if len(numeric_cols) >= 2:
            st.plotly_chart(px.line(df[numeric_cols]), use_container_width=True)
            return

    if chart_type == "Bar" and len(df.columns) >= 2:
        st.plotly_chart(px.bar(df, x=df.columns[0], y=df.columns[-1]), use_container_width=True)
    elif chart_type == "Line" and len(df.columns) >= 2:
        st.plotly_chart(px.line(df, x=df.columns[0], y=df.columns[-1]), use_container_width=True)
    elif chart_type == "Pie" and len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        st.plotly_chart(px.pie(df, names=cat_cols[0], values=numeric_cols[0]), use_container_width=True)
    elif chart_type == "Scatter" and len(numeric_cols) >= 2:
        st.plotly_chart(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1]), use_container_width=True)


# ───────────────────────────────────────────────
# SIDEBAR – DATABASE CONNECTION
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("🗄 Database Connection")
    db_name = st.text_input("SQLite filename", value="healthcare.db", help="e.g. healthcare.db")

    if st.button("Connect to Database", type="primary"):
        try:
            engine = create_engine(f"sqlite:///{db_name}")
            pd.read_sql("SELECT 1", engine)  # quick test
            st.session_state.engine = engine
            st.success("Connected!", icon="✅")
        except Exception as e:
            st.session_state.engine = None
            st.error(f"Connection failed: {str(e)}")

    st.divider()
    st.header("Navigation")
    page = st.radio(
        "Select module",
        ["Home", "SQL Assistant", "Patient Reports", "Database Explorer"],
        format_func=lambda x: f"🏠 {x}" if x == "Home" else f"🧠 {x}" if x == "SQL Assistant" else f"📄 {x}" if x == "Patient Reports" else f"📊 {x}"
    )


# ───────────────────────────────────────────────
# PAGES
# ───────────────────────────────────────────────
engine = st.session_state.get("engine")

if page == "Home":
    st.title("🏥 Clinical Data Assistant")
    st.markdown("""
    An intelligent healthcare companion that helps you:

    • Ask natural language questions about your patient database  
    • Get instant SQL-powered insights & visualizations  
    • Analyze uploaded medical reports with clinical reasoning  
    • Explore raw database tables safely

    **Select a module from the sidebar to begin.**
    """)

    st.info("Connect your SQLite database in the sidebar to unlock SQL Assistant and Database Explorer.", icon="ℹ️")


elif page == "SQL Assistant":
    st.title("🧠 Natural Language → SQL Insights")
    if not engine:
        st.warning("Please connect to a SQLite database first using the sidebar.", icon="⚠️")
        st.stop()

    # Initialize history
    if "sql_history" not in st.session_state:
        st.session_state.sql_history = []

    # Suggested questions
    st.subheader("Quick Start Questions")
    sample_questions = [
        "How many patients are there?",
        "Show all female patients over age 60",
        "Which patient has the most medications?",
        "Count visits per reason in the last year",
        "Average age by gender"
    ]

    cols = st.columns(5)
    for i, q in enumerate(sample_questions):
        if cols[i % 5].button(q, key=f"sample_sql_{i}", use_container_width=True):
            with st.spinner("Generating & executing query..."):
                try:
                    sql = generate_sql(q)
                    if sql:
                        df = pd.read_sql(text(sql), engine)
                        st.session_state.sql_history.insert(0, {"question": q, "sql": sql, "df": df})
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.divider()

    # User question
    user_question = st.chat_input("Ask anything about your patient database...")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                sql = generate_sql(user_question)
                if sql:
                    df = pd.read_sql(text(sql), engine)
                    st.session_state.sql_history.insert(0, {"question": user_question, "sql": sql, "df": df})
                    st.rerun()
            except Exception as e:
                st.error(f"Could not process query: {str(e)}")

    # History – newest first
    if st.session_state.sql_history:
        st.subheader("Query History")

        if st.button("Clear history", type="secondary"):
            st.session_state.sql_history = []
            st.rerun()

        for item in st.session_state.sql_history:
            with st.expander(f"❓ {item['question']}", expanded=(item == st.session_state.sql_history[0])):
                tabs = st.tabs(["📊 Result", "📈 Chart", "🧾 SQL"])

                with tabs[0]:
                    st.dataframe(item["df"], use_container_width=True)

                with tabs[1]:
                    render_auto_chart(item["df"])

                with tabs[2]:
                    st.code(item["sql"], language="sql")


elif page == "Patient Reports":
    st.title("📄 Clinical Report Intelligence")
    st.caption("Upload PDF reports and ask clinically relevant questions")

    col_left, col_right = st.columns([5, 3])

    with col_left:
        pdf = st.file_uploader(
            "Upload medical report (PDF)",
            type="pdf",
            help="Discharge summary, echo, cath report, labs, imaging, etc."
        )

    with col_right:
        patient_id = st.text_input(
            "Patient identifier",
            placeholder="e.g. Manish Singh, PID-3742, Mr. Patel",
            help="Used to separate different patients' documents"
        ).strip()

    if pdf and patient_id:
        success, msg = process_and_index_report(pdf, patient_id)
        if success:
            st.success(msg, icon="✅")
        else:
            st.error(msg)

    # Show currently active patient
    if patient_id and patient_id in st.session_state.patient_docs:
        info = st.session_state.patient_docs[patient_id]
        st.caption(f"Active report: **{info['filename']}** • loaded {info['loaded_at']}")

    question = st.chat_input("Ask a clinical question about this report...")

    if question and patient_id:
        if patient_id not in st.session_state.patient_docs:
            st.warning("Please upload and process a report for this patient first.")
        else:
            with st.chat_message("assistant"):
                answer, err = query_report(question, patient_id)
                if err:
                    st.error(err)
                elif answer:
                    st.markdown(answer)
                else:
                    st.info("No answer could be generated from the report.")


elif page == "Database Explorer":
    st.title("📊 Database Explorer")
    if not engine:
        st.warning("Connect to database first via sidebar.")
        st.stop()

    try:
        tables_df = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            engine
        )
        tables = tables_df["name"].tolist()

        if not tables:
            st.info("No tables found in database.")
        else:
            selected_table = st.selectbox("Select table to view", tables)

            if selected_table:
                df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 1500", engine)
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(df)} rows shown (max 1500)")

    except Exception as e:
        st.error(f"Error reading database: {str(e)}")
