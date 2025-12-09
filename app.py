import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from sqlalchemy import create_engine, text

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

# --- Page Configuration ---
st.set_page_config(
    page_title="Healthcare NL→SQL",
    page_icon="🏥",
    layout="wide"
)

# --- UI Header ---
st.markdown("""
    <style>
    .header {
        background-color: #2e3b4e;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header h1 {
        font-size: 2.5em;
        margin: 0;
    }
    .header p {
        font-size: 1.1em;
        margin-top: 10px;
    }
    </style>
    <div class="header">
        <h1>🏥 Healthcare: Natural Language to SQL</h1>
        <p>Your AI-powered assistant for clinical research data analysis.</p>
    </div>
""", unsafe_allow_html=True)


# ------------------ Sidebar Content (No API key input needed) ------------------
with st.sidebar:
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses AI to translate plain English questions "
        "into SQL queries, allowing clinical researchers to interact with "
        "patient data intuitively and in real-time."
    )
    
    st.sidebar.title("How to Use")
    st.sidebar.markdown(
        """
        1.  Ask a question about the data in the chat box.
        2.  View the results in the organized tabs.
        3.  Click "Clear message history" to start over.
        """
    )

# ------------------ LLM Configuration ------------------
# MODIFIED: API key is now loaded securely from st.secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Groq API key not found. Please add it to your .streamlit/secrets.toml file.")
    st.stop()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=api_key,
    max_tokens=2048
)

# ------------------ Database Configuration ------------------
@st.cache_resource
def get_db():
    db_filepath = (Path(__file__).parent / "healthcare.db").absolute()
    if not db_filepath.exists():
        st.error(f"Database file not found at: {db_filepath}")
        st.stop()
    creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
    engine = create_engine("sqlite:///", creator=creator)
    custom_table_info = {
        "patients": "Contains demographic information about each patient (id, name, age, gender). The gender column uses 'M' for male and 'F' for female.",
        "visits": "Records patient visits, including the date and reason for the visit. Links to patients by patient_id.",
        "medications": "Lists medications prescribed to patients. Links to patients by patient_id. The column is named 'medication'."
    }
    return SQLDatabase(engine, custom_table_info=custom_table_info)

try:
    db = get_db()
except Exception as e:
    st.error(f"Failed to connect to the database: {str(e)}")
    st.stop()

# ------------------ Prompt Template ------------------
# MODIFIED: Added instruction to use aliases and updated examples.
PROMPT_TEMPLATE = """
You are an expert SQLite data analyst. Your task is to convert a user's question in plain English into a valid SQLite query.

**Instructions:**
1.  You must only output the SQL query. Do not include any other text, explanations, or markdown.
2.  When using aggregate functions (COUNT, AVG, etc.), always give the calculated column a simple, clear alias using `AS`. For example, `COUNT(*) AS total_count`.
3.  Analyze the database schema below to understand the available tables and columns.
4.  Pay close attention to the user's question to identify the key entities and relationships required for the query.
5.  Use the provided few-shot examples as a guide for constructing correct queries.

**Database Schema:**
{table_info}

**Few-shot Examples:**
-- Question: How many patients are there for each gender?
-- SQL: SELECT gender, COUNT(*) AS patient_count FROM patients GROUP BY gender;

-- Question: Show me all the male patients.
-- SQL: SELECT * FROM patients WHERE gender = 'M';

-- Question: What was the reason for Alice Smith's latest visit?
-- SQL: SELECT T2.reason FROM patients AS T1 JOIN visits AS T2 ON T1.id = T2.patient_id WHERE T1.name = 'Alice Smith' ORDER BY T2.visit_date DESC LIMIT 1;

-- Question: Show the number of medications per patient.
-- SQL: SELECT T1.name, COUNT(T2.medication) AS medication_count FROM patients AS T1 JOIN medications AS T2 ON T1.id = T2.patient_id GROUP BY T1.name;

**User Question:**
{question}

**SQL Query:**
"""

prompt = PromptTemplate(
    input_variables=["question", "table_info"],
    template=PROMPT_TEMPLATE,
)

# ------------------ LangChain Execution Chain ------------------
generate_query_chain = (
    RunnablePassthrough.assign(table_info=lambda x: db.get_table_info())
    | prompt
    | llm
    | StrOutputParser()
)

# --- Reusable function to display results in tabs ---
def display_results(df, sql_query):
    with st.container(border=True):
        tab_table, tab_chart, tab_sql = st.tabs(["📊 Table", "📈 Chart", "🔍 SQL"])
        
        with tab_table:
            st.dataframe(df, use_container_width=True)
        
        with tab_sql:
            st.code(sql_query, language="sql")
        
        with tab_chart:
            try:
                if df.shape == (1, 1):
                    st.metric("Result", df.iloc[0, 0])
                elif len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                    st.bar_chart(df, x=df.columns[0], y=df.columns[1])
                else:
                    st.info("A chart can't be automatically generated for this data.")
            except Exception as e:
                st.warning(f"Could not generate chart: {e}")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! Ask a question about the healthcare data or try one of the examples below."}]

# --- Example Questions ---
example_questions = [
    "How many patients are there?",
    "Which patients are on the most medications?",
    "Show me all the male patients.",
    "What was the reason for Alice Smith's latest visit?"
]

def set_query_from_example(question):
    st.session_state.user_query_from_button = question

st.write("### Example Questions")
cols = st.columns(len(example_questions))
for i, question in enumerate(example_questions):
    with cols[i]:
        if st.button(question, use_container_width=True, on_click=set_query_from_example, args=[question]):
            pass

st.markdown("---")

# ------------------ Chat UI ------------------
# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], dict) and "is_result" in msg["content"]:
            display_results(msg["content"]["df"], msg["content"]["sql"])
        else:
            st.write(msg["content"])

# Handle new user input
if user_query := st.chat_input(placeholder="Ask about patients, visits, or medications...", key="main_chat_input") or st.session_state.get("user_query_from_button"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                sql_query = generate_query_chain.invoke({"question": user_query})
                engine = db._engine
                
                with engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    columns = result.keys()
                    rows = result.fetchall()

                if rows:
                    df = pd.DataFrame(rows, columns=columns)
                    result_data = {"is_result": True, "df": df, "sql": sql_query}
                    st.session_state.messages.append({"role": "assistant", "content": result_data})
                    display_results(df, sql_query)
                else:
                    response = "The query returned no results."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(error_message)
    
    if "user_query_from_button" in st.session_state:
        del st.session_state["user_query_from_button"]