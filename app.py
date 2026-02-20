import streamlit as st
import pandas as pd
import sqlite3
import json
import requests
import pdfplumber
from pathlib import Path
from sqlalchemy import create_engine, text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

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

db_path = Path(__file__).parent / "demo_healthcare.db"
engine = create_engine(f"sqlite:///{db_path}")

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
You are an expert SQL generator for a SQLite healthcare database.

STRICT RULES:
- Output ONLY valid SQL.
- No explanations.
- No markdown, no backticks.
- Never invent tables or columns.
- 'M' = Male, 'F' = Female.
- Use COUNT(*) AS alias for counts.

SCHEMA:
patients(id, name, age, gender)
visits(id, patient_id, visit_date, reason)
medications(id, patient_id, medication)

USER QUESTION:
{question}

SQL:
"""

def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    sql = call_llm([{"role": "user", "content": prompt}])
    return sql.replace("```","").strip()

# -------------------------------------------------------
# RAG SETUP
# -------------------------------------------------------
@st.cache_resource
def get_vector_db():
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint="https://makeathonmj-ai.openai.azure.com/",
        api_key=API_KEY,
        azure_deployment="text-embedding-3-small"
    )
    return Chroma(
        collection_name="reports",
        embedding_function=embeddings,
        persist_directory="chroma_reports"
    )

vector_db = get_vector_db()

def extract_pdf(pdf):
    with pdfplumber.open(pdf) as p:
        return "\n".join([pg.extract_text() for pg in p.pages if pg.extract_text()])

def embed_report(text, patient):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(text)
    vector_db.add_texts(chunks, metadatas=[{"patient": patient}] * len(chunks))
    vector_db.persist()

def answer_report(question, patient):
    docs = vector_db.similarity_search(question, k=3, filter={"patient": patient})
    if not docs:
        return "No report data found."

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer clinically:"
    return call_llm([{"role":"user","content": prompt}])

# -------------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🧠 SQL Assistant", "📄 Patient RAG", "⚙️ Database Viewer"]
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    <div style="background:#1F2937;padding:35px;border-radius:15px;text-align:center">
        <h1 style="color:white;">🏥 Healthcare AI Assistant</h1>
        <p style="color:#d1d5db;font-size:18px;">
            Your all-in-one platform for Healthcare Data Intelligence 🚀  
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Welcome!")
    st.write("Use the sidebar to navigate between modules:")

    st.info("""
    **🧠 SQL Assistant** → Ask natural questions about your database  
    **📄 Patient RAG** → Upload PDFs & ask patient-specific clinical questions  
    **⚙️ DB Viewer** → View your SQLite tables  
    """)

# -------------------------------------------------------
# SQL ASSISTANT
# -------------------------------------------------------
elif page == "🧠 SQL Assistant":
    st.title("🧠 Natural Language → SQL Assistant")

    if "history" not in st.session_state:
        st.session_state.history = []

    # ---------------------------
    # EXAMPLE QUESTIONS
    # ---------------------------
    st.write("### 🔍 Try an Example Question")

    examples = [
        "How many patients are there?",
        "Show all male patients.",
        "How many male and female patients?",
        "Which patient has the most medications?",
        "List all visits by date.",
        "Show patient count grouped by age.",
        "What is the average age of patients?"
    ]

    cols = st.columns(3)
    for i, q in enumerate(examples):
        with cols[i % 3]:
            if st.button(q):
                try:
                    sql = generate_sql(q)
                    with engine.connect() as conn:
                        df = pd.read_sql(text(sql), conn)

                    st.session_state.history.append({"q": q, "sql": sql, "df": df})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")

    # ---------------------------
    # USER QUESTION
    # ---------------------------
    question = st.chat_input("Ask a question about your healthcare data...")

    if question:
        try:
            sql = generate_sql(question)
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)

            st.session_state.history.append({"q": question, "sql": sql, "df": df})
            st.rerun()
        except Exception as e:
            st.error(str(e))

    # --------------------------------------------------------
    # SMART CHART ENGINE
    # --------------------------------------------------------
    import plotly.express as px

def auto_chart(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cols = df.columns

    # Title helpers
    def title_from_cols(x, y=None):
        if y:
            return f"{y.replace('_',' ').title()} by {x.replace('_',' ').title()}"
        return x.replace('_',' ').title()

    # 1) Single metric → show metric
    if df.shape == (1, 1):
        st.metric(df.columns[0], df.iloc[0, 0])
        return

    # 2) User chart override selector
    chart_choice = st.selectbox(
        "Choose visualization:",
        ["Auto", "Bar Chart", "Line Chart", "Pie Chart", "Scatter"],
        key=str(cols)
    )

    # --- AUTO MODE ---
    if chart_choice == "Auto":

        # A) Date column → Line chart
        date_cols = [c for c in cols if "date" in c.lower()]
        if date_cols and len(numeric_cols) >= 1:
            fig = px.line(
                df, x=date_cols[0], y=numeric_cols,
                title=title_from_cols(date_cols[0])
            )
            st.plotly_chart(fig, use_container_width=True)
            return

        # B) Text category + numeric → Bar chart
        if len(text_cols) == 1 and len(numeric_cols) == 1:
            x, y = text_cols[0], numeric_cols[0]
            fig = px.bar(df, x=x, y=y, title=title_from_cols(x, y))
            st.plotly_chart(fig, use_container_width=True)
            return

        # C) Numeric discrete categories → Bar chart
        if len(numeric_cols) == 2:
            x, y = numeric_cols[0], numeric_cols[1]
            if df[x].nunique() <= 50:   # the age fix
                fig = px.bar(df, x=x, y=y, title=title_from_cols(x, y))
                st.plotly_chart(fig, use_container_width=True)
                return

        # D) Multiple numeric → Line chart
        if len(numeric_cols) >= 2:
            fig = px.line(df, y=numeric_cols, title="Numeric Trends")
            st.plotly_chart(fig, use_container_width=True)
            return

        # E) Percentages → Pie
        if any("%" in c.lower() for c in cols):
            x, y = text_cols[0], numeric_cols[0]
            fig = px.pie(df, values=y, names=x, title=title_from_cols(x, y))
            st.plotly_chart(fig, use_container_width=True)
            return

        st.info("No suitable chart found.")
        return

    # --- USER OVERRIDE MODES (if they choose manually) ---
    if chart_choice == "Bar Chart":
        if len(numeric_cols) >= 1:
            fig = px.bar(df, x=cols[0], y=numeric_cols[-1], title=title_from_cols(cols[0], numeric_cols[-1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric data for Bar Chart.")
        return

    if chart_choice == "Line Chart":
        if len(numeric_cols) >= 1:
            fig = px.line(df, x=cols[0], y=numeric_cols, title=title_from_cols(cols[0]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric data for Line Chart.")
        return

    if chart_choice == "Pie Chart":
        if len(text_cols) >= 1 and len(numeric_cols) >= 1:
            fig = px.pie(df, names=text_cols[0], values=numeric_cols[-1], title=title_from_cols(text_cols[0], numeric_cols[-1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need 1 category & 1 numeric column.")
        return

    if chart_choice == "Scatter":
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=title_from_cols(numeric_cols[0], numeric_cols[1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need two numeric columns.")
        return
    def auto_chart(df):
        import pandas as pd
        import streamlit as st

        # 1) Single metric
        if df.shape == (1, 1):
            st.metric("Result", df.iloc[0, 0])
            return

        cols = df.columns

        # Identify types
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        text_cols = df.select_dtypes(include='object').columns.tolist()
        date_cols = [c for c in cols if "date" in c.lower()]

        # 2) Line chart if a date column exists
        if date_cols and len(numeric_cols) >= 1:
            st.line_chart(df, x=date_cols[0], y=numeric_cols)
            return
        
        # 3) Category + numeric → bar chart
        if len(text_cols) == 1 and len(numeric_cols) == 1:
            st.bar_chart(df, x=text_cols[0], y=numeric_cols[0])
            return

        # 4) Multiple numeric columns → line chart
        if len(numeric_cols) >= 2:
            st.line_chart(df[numeric_cols])
            return

        # 5) Percentage columns → pie chart
        if any("%" in c.lower() or "rate" in c.lower() for c in cols):
            import plotly.express as px
            cname = numeric_cols[0]
            fig = px.pie(df, values=cname, names=text_cols[0])
            st.plotly_chart(fig)
            return

        # 6) Fallback
        st.info("No suitable visualization for this query.")

    # --------------------------------------------------------
    # DISPLAY HISTORY
    # --------------------------------------------------------
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
        pdf = st.file_uploader("Upload PDF report", type=["pdf"])
    with col2:
        patient = st.text_input("Patient Name")

    if pdf and patient:
        text = extract_pdf(pdf)
        embed_report(text, patient)
        st.success("Report embedded successfully!")

    q = st.text_input("Ask something about this patient’s report")

    if q and patient:
        with st.spinner("Analyzing..."):
            ans = answer_report(q, patient)
        st.write("### 🧠 Clinical Insight")
        st.write(ans)

# -------------------------------------------------------
# DATABASE VIEWER
# -------------------------------------------------------
elif page == "⚙️ Database Viewer":
    st.title("⚙️ Database Viewer")

    tables = ["patients", "visits", "medications"]

    for t in tables:
        st.subheader(f"📌 {t}")
        df = pd.read_sql(f"SELECT * FROM {t}", engine)
        st.dataframe(df, use_container_width=True)

can you gve complete ready code
