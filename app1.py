import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# ------------------ Database Options ------------------
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite 3 Database - student.db", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB to chat with", options=radio_opt)

db_uri = LOCALDB if radio_opt.index(selected_opt) == 0 else MYSQL

mysql_host = mysql_user = mysql_password = mysql_db = None
if db_uri == MYSQL:
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")

# ------------------ Groq API Key ------------------
api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not api_key:
    st.info("Please add the Groq API key.")
    st.stop()

# ✅ Use supported Groq model
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # You can switch to "llama-3.3-70b-versatile"
    temperature=0,
    api_key=api_key
)

# ------------------ DB Config ------------------
def configure_db():
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "students.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}", uri=True)
        engine = create_engine("sqlite:///", creator=creator)
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        engine = create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        )
    return SQLDatabase(engine)

# ------------------ Agent Setup ------------------
try:
    db = configure_db()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
except Exception as e:
    st.error(f"Failed to connect to the database: {str(e)}")
    st.stop()

# ------------------ Chat UI ------------------
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
