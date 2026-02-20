import streamlit as st
import pandas as pd
import json
import requests
from pathlib import Path
from sqlalchemy import create_engine, text
import urllib

# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Keep all Q/A + results
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------
# AZURE OPENAI CONFIG
# -------------------------------------------------------
API_ENDPOINT = (
    "https://makeathonmj-ai.openai.azure.com/"
    "openai/deployments/gpt-4o-mini-deploy/chat/completions"
    "?api-version=2025-01-01-preview"
)

API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]

def call_llm(messages):
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    payload = {
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1500
    }
    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"Azure Error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# -------------------------------------------------------
# DATABASE SELECTION (SQL SERVER / SQLITE)
# -------------------------------------------------------
st.sidebar.title("Database Configuration")

db_type = st.sidebar.radio("Select Database", ["SQLite (Local)", "SQL Server"])

engine = None

### SQLITE MODE
if db_type == "SQLite (Local)":
    db_path = Path(__file__).parent / "demo_healthcare.db"
    engine = create_engine(f"sqlite:///{db_path}")

### SQL SERVER MODE
else:
    server = st.sidebar.text_input("Server")
    database = st.sidebar.text_input("Database")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Connect"):
        try:
            conn_str = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                "TrustServerCertificate=yes;"
            )
            params = urllib.parse.quote_plus(conn_str)
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
            st.sidebar.success("Connected Successfully!")
        except Exception as e:
            st.sidebar.error(str(e))

# Stop app if engine not ready
if engine is None:
    st.stop()


# -------------------------------------------------------
# SQL GENERATION PROMPT — FINAL VERSION
# -------------------------------------------------------
SQL_PROMPT = """
You are an expert SQL Server query generator for a healthcare database.

STRICT RULES:
- Output ONLY valid SQL.
- No markdown, no backticks, no explanation.
- Never invent tables or columns.
- Use only the following schema:

dbo.patients(id, name, age, gender, city, chronic_condition)
dbo.visits(id, patient_id, visit_date, department, reason)
dbo.medications(id, patient_id, medication, dosage)
dbo.lab_results(id, patient_id, test_name, result_value, unit)

GENDER:
'M' = Male
'F' = Female

RULES:
- Use aliases: P = patients, V = visits, M = medications, L = lab_results
- Use COUNT(*) AS alias for counts
- Use GROUP BY when aggregation exists
- When the question asks for "list", return SELECT rows
- When the question asks for "how many", return a COUNT

EXAMPLES:

Q: How many male and female patients?
SQL:
SELECT gender, COUNT(*) AS patient_count
FROM dbo.patients
GROUP BY gender;

Q: Show all male patients.
SQL:
SELECT *
FROM dbo.patients
WHERE gender = 'M';

Q: Which patient has most medications?
SQL:
SELECT P.name, COUNT(M.medication) AS medication_count
FROM dbo.patients AS P
JOIN dbo.medications AS M ON P.id = M.patient_id
GROUP BY P.name
ORDER BY medication_count DESC;

Q: List lab results for patient John Doe.
SQL:
SELECT L.test_name, L.result_value, L.unit
FROM dbo.lab_results AS L
JOIN dbo.patients AS P ON P.id = L.patient_id
WHERE P.name = 'John Doe';

USER QUESTION:
{question}

SQL:
"""

def generate_sql(question):
    prompt = SQL_PROMPT.format(question=question)
    raw_sql = call_llm([{"role": "user", "content": prompt}]).strip()

    sql = raw_sql.replace("```", "").replace("`", "").strip()

    # Block unsafe SQL
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", ";--"]
    if any(word in sql.upper() for word in blocked):
        raise Exception("Unsafe SQL detected! Not allowed.")

    return sql


# -------------------------------------------------------
# HEADER UI
# -------------------------------------------------------
st.markdown("""
<div style="background:#1F2937;padding:25px;border-radius:12px;text-align:center">
<h2 style="color:white">🏥 Healthcare AI Assistant</h2>
<p style="color:#cbd5e1">Natural Language → SQL → Insights</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# EXAMPLE QUESTIONS
# -------------------------------------------------------
st.write("### Example Questions")

examples = [
    "How many patients are there?",
    "How many male and female patients?",
    "Show all male patients.",
    "Which patient has the highest number of visits?",
    "List lab results for patient Alice Smith."
]

cols = st.columns(len(examples))

for i, q in enumerate(examples):
    with cols[i]:
        if st.button(q, use_container_width=True):
            try:
                sql = generate_sql(q)
                with engine.connect() as conn:
                    df = pd.read_sql(text(sql), conn)

                st.session_state.history.append({
                    "question": q,
                    "sql": sql,
                    "df": df
                })
                st.rerun()

            except Exception as e:
                st.error(str(e))

st.markdown("---")

# -------------------------------------------------------
# CHAT INPUT
# -------------------------------------------------------
user_q = st.chat_input("Ask a question about your data...")

if user_q:
    try:
        sql = generate_sql(user_q)

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)

        st.session_state.history.append({
            "question": user_q,
            "sql": sql,
            "df": df
        })

        st.rerun()

    except Exception as e:
        st.error(str(e))


# -------------------------------------------------------
# DISPLAY FULL HISTORY (ALL RESULTS STACKED)
# -------------------------------------------------------
for item in st.session_state.history:

    with st.container(border=True):

        st.markdown(f"### ❓ {item['question']}")
        df = item["df"]
        sql = item["sql"]

        res_tab, chart_tab, sql_tab = st.tabs(["📊 Result", "📈 Chart", "🧾 SQL"])

        with res_tab:
            if df.shape == (1, 1):
                st.metric("Result", df.iloc[0, 0])
            else:
                st.dataframe(df, use_container_width=True)

        with chart_tab:
            if df.shape == (1, 1):
                st.info("Single value — no chart.")
            elif len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                st.bar_chart(df, x=df.columns[0], y=df.columns[1])
            elif "date" in df.columns[0].lower():
                st.line_chart(df, x=df.columns[0], y=df.columns[1])
            else:
                st.info("No suitable chart detected.")

        with sql_tab:
            st.code(sql, language="sql")
