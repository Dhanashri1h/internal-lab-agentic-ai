"""
Experiment 4: SQL Agent with Tool Use (ReAct)
Applied Agentic AI Lab
"""
import os
import sqlite3

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

DB_PATH = "company.db"


# ---------------------------------------------------------------
# STEP 1: Create a sample SQLite database
# ---------------------------------------------------------------
def setup_database(path: str = DB_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE departments (
        dept_id   INTEGER PRIMARY KEY,
        dept_name TEXT NOT NULL,
        location  TEXT
    );
    CREATE TABLE employees (
        emp_id    INTEGER PRIMARY KEY,
        name      TEXT NOT NULL,
        dept_id   INTEGER REFERENCES departments(dept_id),
        salary    REAL,
        hire_date TEXT
    );
    CREATE TABLE projects (
        project_id INTEGER PRIMARY KEY,
        title      TEXT NOT NULL,
        dept_id    INTEGER REFERENCES departments(dept_id),
        budget     REAL
    );
    """)
    cur.executemany("INSERT INTO departments VALUES (?,?,?)", [
        (1, "Engineering", "Hyderabad"),
        (2, "HR", "Mumbai"),
        (3, "Sales", "Delhi"),
    ])
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", [
        (1, "Asha",   1, 90000, "2021-03-15"),
        (2, "Ravi",   1, 85000, "2020-07-01"),
        (3, "Meena",  2, 60000, "2022-01-10"),
        (4, "Karthik",3, 70000, "2019-11-23"),
        (5, "Sana",   1, 95000, "2018-05-30"),
        (6, "Vikram", 3, 65000, "2023-02-14"),
    ])
    cur.executemany("INSERT INTO projects VALUES (?,?,?,?)", [
        (1, "AI Chatbot",   1, 500000),
        (2, "Hiring Portal",2, 150000),
        (3, "CRM Upgrade",  3, 300000),
    ])
    con.commit()
    con.close()


# ---------------------------------------------------------------
# STEP 2: Define the database TOOLS the agent can call
# ---------------------------------------------------------------
def _connect():
    return sqlite3.connect(DB_PATH)


@tool
def list_tables() -> str:
    """List all table names in the database. Always call this first."""
    con = _connect()
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'").fetchall()
    con.close()
    return ", ".join(r[0] for r in rows)


@tool
def describe_table(table_name: str) -> str:
    """Return the columns (name, type) and 3 sample rows of a table."""
    con = _connect()
    try:
        cols = con.execute(f"PRAGMA table_info({table_name})").fetchall()
        if not cols:
            return f"Error: table '{table_name}' does not exist."
        schema = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
        return f"Table {table_name}({schema})\nSample rows: {sample}"
    finally:
        con.close()


@tool
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL SELECT query and return the rows.
    If the query fails, an error message is returned so you can fix it."""
    q = query.strip().rstrip(";")
    if not q.lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    con = _connect()
    try:
        cur = con.execute(q)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(50)
        return f"Columns: {cols}\nRows: {rows}"
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        con.close()


TOOLS = [list_tables, describe_table, run_sql_query]


# ---------------------------------------------------------------
# STEP 3: Choose an LLM (set the matching API key as env variable)
# ---------------------------------------------------------------
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":      # needs ANTHROPIC_API_KEY
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    if provider == "groq":           # needs GROQ_API_KEY (free tier)
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    if provider == "google":         # needs GOOGLE_API_KEY (free tier)
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    if provider == "openai":         # needs OPENAI_API_KEY
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raise ValueError("Unknown LLM_PROVIDER")


# ---------------------------------------------------------------
# STEP 4: Build the ReAct agent (Reason -> Act -> Observe loop)
# ---------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful SQL assistant for a SQLite database.
Follow the ReAct pattern: think about what you need, call a tool, read the
observation, and repeat until you can answer.
Rules:
1. Always call list_tables first, then describe_table for relevant tables.
2. Never guess column names. Use only what describe_table shows.
3. Write only SELECT queries. Never modify data.
4. If run_sql_query returns an error, read it, fix the query and retry.
5. Give a short final answer in plain English based on the query result."""


def build_agent():
    return create_react_agent(get_llm(), TOOLS, prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------
# STEP 5: Run the agent and print every Thought/Action/Observation
# ---------------------------------------------------------------
def ask(agent, question: str) -> None:
    print("=" * 70)
    print("QUESTION:", question)
    print("=" * 70)
    for step in agent.stream({"messages": [("user", question)]},
                             stream_mode="values"):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    setup_database()
    agent = build_agent()
    ask(agent, "how many employees are there ?")
    ask(agent, "List employees in Engineering earning more than 88000.")
    ask(agent, "What is the total project budget per department location?")
