"""
Experiment 2: Text-to-SQL - End-to-End LLM Workflow
Applied Agentic AI Lab

Pipeline (5 stages):
  Question -> [1] RETRIEVE schema context -> [2] GENERATE SQL (LLM)
           -> [3] VALIDATE / REPAIR       -> [4] EXECUTE on database
           -> [5] EXPLAIN result (LLM)    -> SAVE files
"""
import csv
import math
import os
import re
import sqlite3
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

DB_PATH = "company.db"
OUT_DIR = "outputs_exp2"     # all generated files go here
TOP_K = 4                    # how many knowledge-base chunks to retrieve
MAX_RETRIES = 2              # how many times the LLM may repair a bad query


# ==============================================================
# PART 1: DATABASE
# ==============================================================
def setup_database(path: str = DB_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE departments (dept_id INTEGER PRIMARY KEY, dept_name TEXT NOT NULL, location TEXT);
    CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
        dept_id INTEGER REFERENCES departments(dept_id), salary REAL, hire_date TEXT);
    CREATE TABLE projects (project_id INTEGER PRIMARY KEY, title TEXT NOT NULL,
        dept_id INTEGER REFERENCES departments(dept_id), budget REAL);
    """)
    cur.executemany("INSERT INTO departments VALUES (?,?,?)",
                    [(1, "Engineering", "Hyderabad"), (2, "HR", "Mumbai"), (3, "Sales", "Delhi")])
    cur.executemany("INSERT INTO employees VALUES (?,?,?,?,?)", [
        (1, "Asha", 1, 90000, "2021-03-15"), (2, "Ravi", 1, 85000, "2020-07-01"),
        (3, "Meena", 2, 60000, "2022-01-10"), (4, "Karthik", 3, 70000, "2019-11-23"),
        (5, "Sana", 1, 95000, "2018-05-30"), (6, "Vikram", 3, 65000, "2023-02-14")])
    cur.executemany("INSERT INTO projects VALUES (?,?,?,?)",
                    [(1, "AI Chatbot", 1, 500000), (2, "Hiring Portal", 2, 150000),
                     (3, "CRM Upgrade", 3, 300000)])
    con.commit()
    con.close()


# ==============================================================
# PART 2: KNOWLEDGE BASE  (this is what we RETRIEVE from)
# Table descriptions + relationships + example question->SQL pairs.
# Add your own entries here to improve the workflow.
# ==============================================================
KNOWLEDGE_BASE = [
    {"type": "schema", "text":
        "Table departments(dept_id INTEGER PRIMARY KEY, dept_name TEXT, location TEXT). "
        "Stores each department and the city (location) where it is based."},
    {"type": "schema", "text":
        "Table employees(emp_id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary REAL, hire_date TEXT). "
        "Stores staff members, their pay (salary, earning, income), and hire date (joined, hired, recent). "
        "hire_date format is YYYY-MM-DD."},
    {"type": "schema", "text":
        "Table projects(project_id INTEGER PRIMARY KEY, title TEXT, dept_id INTEGER, budget REAL). "
        "Stores company projects, the department that owns each project, and the project budget (cost, money)."},
    {"type": "relationship", "text":
        "Relationships: employees.dept_id = departments.dept_id and projects.dept_id = departments.dept_id. "
        "Join employees or projects to departments to get department name or location (city)."},
    {"type": "example", "text":
        "Question: How many employees are in each department? "
        "SQL: SELECT d.dept_name, COUNT(*) AS total FROM employees e JOIN departments d "
        "ON e.dept_id = d.dept_id GROUP BY d.dept_name"},
    {"type": "example", "text":
        "Question: Which employee has the highest salary? "
        "SQL: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"},
    {"type": "example", "text":
        "Question: What is the total project budget per department? "
        "SQL: SELECT d.dept_name, SUM(p.budget) AS total_budget FROM projects p JOIN departments d "
        "ON p.dept_id = d.dept_id GROUP BY d.dept_name"},
]


# ==============================================================
# PART 3: RETRIEVER  (TF-IDF + cosine similarity, pure Python)
# Runs locally, so it costs no API requests.
# ==============================================================
def tokenize(text: str):
    return re.findall(r"[a-z0-9_]+", text.lower())


class Retriever:
    def __init__(self, docs):
        self.docs = docs
        tokens = [tokenize(d["text"]) for d in docs]
        df = Counter()
        for t in tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self._vector(t) for t in tokens]

    def _vector(self, tokens):
        if not tokens:
            return {}
        tf = Counter(tokens)
        v = {w: (c / len(tokens)) * self.idf[w] for w, c in tf.items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, query: str, k: int = TOP_K):
        qv = self._vector(tokenize(query))
        scored = [(sum(qv.get(w, 0) * x for w, x in dv.items()), d)
                  for dv, d in zip(self.vecs, self.docs)]
        scored.sort(key=lambda s: s[0], reverse=True)
        return scored[:k]


# ==============================================================
# PART 4: LLM  (default = Google Gemini)
# ==============================================================
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":         # needs GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    if provider == "groq":           # needs GROQ_API_KEY
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    if provider == "anthropic":      # needs ANTHROPIC_API_KEY
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    if provider == "openai":         # needs OPENAI_API_KEY
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raise ValueError("Unknown LLM_PROVIDER")


def clean_text(content) -> str:
    """Gemini returns the reply inside a list; keep only the readable text."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def call_llm(llm, prompt: str) -> str:
    return clean_text(llm.invoke(prompt).content).strip()


# ==============================================================
# PART 5: THE WORKFLOW STAGES
# ==============================================================
def build_sql_prompt(question, context, prev_sql=None, error=None):
    p = ("You are an expert SQLite analyst. Using ONLY the context below, write ONE "
         "SQLite SELECT query that answers the question.\n"
         "Return ONLY the SQL query - no explanation and no markdown.\n\n"
         f"CONTEXT:\n{context}\n\nQUESTION: {question}\n")
    if prev_sql:
        p += (f"\nYour previous query was:\n{prev_sql}\n"
              f"It failed with this error: {error}\n"
              "Fix the query and return only the corrected SQL.\n")
    return p


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text, flags=re.I).strip()
    m = re.search(r"\b(select|with)\b", text, flags=re.I)
    sql = text[m.start():] if m else text
    return sql.strip().rstrip(";").strip()


FORBIDDEN = r"\b(insert|update|delete|drop|alter|create|replace|pragma|attach|detach|truncate)\b"


def validate_sql(sql: str):
    """Returns (ok, message). Only single read-only SELECT queries pass."""
    if not re.match(r"^\s*(select|with)\b", sql, flags=re.I):
        return False, "Only SELECT queries are allowed."
    if re.search(FORBIDDEN, sql, flags=re.I):
        return False, "Query contains a forbidden (write) keyword."
    if ";" in sql:
        return False, "Only one statement is allowed."
    try:                                   # dry run: catches wrong table/column names
        con = sqlite3.connect(DB_PATH)
        con.execute("EXPLAIN QUERY PLAN " + sql)
        con.close()
    except Exception as e:
        return False, f"SQL Error: {e}"
    return True, "OK"


def execute_sql(sql: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    con.close()
    return cols, rows


def save_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def run_workflow(llm, retriever, question: str, idx: int) -> dict:
    print("\n" + "=" * 70)
    print(f"QUESTION {idx}: {question}")
    print("=" * 70)
    result = {"question": question, "context": [], "sql": "", "cols": [], "rows": [],
              "answer": "", "attempts": 0, "status": "FAILED", "csv": ""}

    # ---- Stage 1: RETRIEVE ----
    print("\n[1/5] RETRIEVE - finding the most relevant schema context")
    hits = retriever.search(question)
    for score, d in hits:
        print(f"   score={score:.2f}  ({d['type']})  {d['text'][:70]}...")
    context = "\n".join(f"- {d['text']}" for _, d in hits)
    result["context"] = [(round(s, 2), d["type"], d["text"]) for s, d in hits]

    # ---- Stages 2-3: GENERATE + VALIDATE (with repair loop) ----
    sql, error = None, None
    for attempt in range(1, MAX_RETRIES + 2):
        result["attempts"] = attempt
        print(f"\n[2/5] GENERATE SQL (attempt {attempt})")
        sql = extract_sql(call_llm(llm, build_sql_prompt(question, context, sql, error)))
        print("   " + sql.replace("\n", "\n   "))
        print("[3/5] VALIDATE")
        ok, msg = validate_sql(sql)
        print(f"   {'PASSED' if ok else 'REJECTED'}: {msg}")
        result["sql"] = sql
        if ok:
            break
        error = msg
    else:
        print("\n   Could not produce a valid query. Skipping this question.")
        return result

    # ---- Stage 4: EXECUTE ----
    print("\n[4/5] EXECUTE on database")
    cols, rows = execute_sql(sql)
    result["cols"], result["rows"] = cols, rows
    print(f"   Columns: {cols}")
    for r in rows[:10]:
        print(f"   {r}")

    # ---- Stage 5: EXPLAIN ----
    print("\n[5/5] EXPLAIN result in plain English")
    prompt = (f"Question: {question}\nSQL used: {sql}\nColumns: {cols}\nRows: {rows[:20]}\n\n"
              "Answer the question in 1-3 short sentences using only these rows.")
    result["answer"] = call_llm(llm, prompt)
    print("   " + result["answer"])

    # ---- Save CSV ----
    result["csv"] = os.path.join(OUT_DIR, f"result_{idx}.csv")
    save_csv(result["csv"], cols, rows)
    result["status"] = "SUCCESS"
    print(f"\n   Saved: {result['csv']}")
    return result


# ==============================================================
# PART 6: SAVE REPORT FILES
# ==============================================================
def write_report(results):
    with open(os.path.join(OUT_DIR, "generated_queries.sql"), "w", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            f.write(f"-- Q{i}: {r['question']}\n{r['sql']};\n\n")

    with open(os.path.join(OUT_DIR, "workflow_report.md"), "w", encoding="utf-8") as f:
        f.write("# Text-to-SQL Workflow Report\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        ok = sum(r["status"] == "SUCCESS" for r in results)
        f.write(f"Questions answered: {ok}/{len(results)}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"## Q{i}. {r['question']}\n\n")
            f.write(f"**Status:** {r['status']}  (attempts: {r['attempts']})\n\n")
            f.write("**Retrieved context:**\n\n")
            for score, typ, text in r["context"]:
                f.write(f"- ({score}) [{typ}] {text[:110]}...\n")
            f.write(f"\n**Generated SQL:**\n\n```sql\n{r['sql']}\n```\n\n")
            if r["status"] == "SUCCESS":
                f.write(f"**Result columns:** {r['cols']}\n\n")
                for row in r["rows"][:10]:
                    f.write(f"- {row}\n")
                f.write(f"\n**Answer:** {r['answer']}\n\n**CSV file:** {r['csv']}\n\n")


# ==============================================================
# MAIN  -  change the questions below and run again
# ==============================================================
QUESTIONS = [
    "Which department has the highest average salary?",
    "List employees in Engineering earning more than 88000.",
    "What is the total project budget per department location?",
]

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_database()
    llm = get_llm()
    retriever = Retriever(KNOWLEDGE_BASE)
    print(f"Knowledge base: {len(KNOWLEDGE_BASE)} chunks | Files will be saved in ./{OUT_DIR}/")

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        results.append(run_workflow(llm, retriever, q, i))

    # optional: type your own questions (just press Enter to finish)
    while True:
        q = input("\nAsk your own question (or press Enter to finish): ").strip()
        if not q:
            break
        results.append(run_workflow(llm, retriever, q, len(results) + 1))

    write_report(results)
    print("\n" + "=" * 70)
    print(f"DONE. Files saved in ./{OUT_DIR}/ :")
    for name in sorted(os.listdir(OUT_DIR)):
        print("   -", name)
