"""
Prompt Chaining for Summarization - Multi-step Prompt Pipeline
Applied Agentic AI Lab

CHAIN (each step's output becomes the next step's input):
  Text -> [1] EXTRACT key points -> [2] DRAFT summary -> [3] CRITIQUE the draft
       -> [4] REFINE using the critique -> [5] FORMAT (title + TL;DR + summary)

BASELINE (for comparison): Text -> ONE single prompt -> summary
"""
import os
import re
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------- settings you can change and experiment with ----------------
OUT_DIR = "outputs_chain"        # all generated files go here
INPUT_FILE = "input.txt"         # put your own text here to summarize it (optional)
MAX_WORDS = 60                   # length limit for the summary
AUDIENCE = "a first-year college student"
DELAY = float(os.getenv("STEP_DELAY", "3"))   # pause between calls (free-tier friendly)

# ---------------- sample text (used if input.txt does not exist) ----------------
SAMPLE_TEXT = """A city college in Hyderabad recently completed a three-month rooftop farming pilot on its science block. Forty student volunteers turned 120 square metres of unused terrace into a small garden growing tomatoes, spinach and chillies. The team used drip irrigation, which cut water use by about 30 percent compared with hand watering, and installed simple shade nets to protect the plants. In total the garden produced 85 kg of vegetables, most of which were sold to the college canteen at a fair price, with the money going back into the project. The pilot was not without problems. Summer heat damaged some of the spinach, a leaking water tank had to be repaired twice, and the number of active volunteers dropped sharply during exam weeks. Even so, the organisers say the experiment showed that unused rooftops can provide fresh food and hands-on learning at very low cost. The college now plans to extend the project to two more buildings and to start a compost unit that will turn canteen food waste into fertiliser for the plants."""

# words we expect a good summary to keep (simple coverage check, sample text only)
KEY_TERMS = ["120", "85", "30", "drip", "compost", "heat", "volunteer", "canteen"]

# ==============================================================
# THE PROMPTS  (the heart of prompt chaining)
# ==============================================================
P1_EXTRACT = """You are a careful reader. Read the text and list the 5 to 7 most important key points as short bullet points.
Use only facts stated in the text. Do not add anything new.

TEXT:
{text}

KEY POINTS:"""

P2_DRAFT = """Using ONLY the key points below, write a summary of 3 to 4 sentences for {audience}.

KEY POINTS:
{key_points}

SUMMARY:"""

P3_CRITIQUE = """You are a strict editor. Compare the draft summary with the original text.
List (a) important information missing from the draft, (b) any statement not supported by the text, (c) unclear wording.
If the draft is fine, write exactly: No issues.
Be brief: at most 5 bullet points.

ORIGINAL TEXT:
{text}

DRAFT SUMMARY:
{draft}

EDITOR FEEDBACK:"""

P4_REFINE = """Rewrite the draft summary to fix the editor feedback. Keep it under {max_words} words, in plain English for {audience}.
Use only facts from the original text. Return only the improved summary.

ORIGINAL TEXT:
{text}

DRAFT SUMMARY:
{draft}

EDITOR FEEDBACK:
{feedback}

IMPROVED SUMMARY:"""

P5_FORMAT = """Format the summary below.
Return exactly these three lines and nothing else:
Title: <a title of at most 8 words>
TL;DR: <one sentence>
Summary: <the summary, unchanged>

SUMMARY:
{summary}"""

P_BASELINE = """Summarize the following text in under {max_words} words for {audience}.

TEXT:
{text}"""


# ==============================================================
# LLM helpers
# ==============================================================
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":         # needs GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
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
    try:
        text = clean_text(llm.invoke(prompt).content).strip()
    except Exception as e:                         # friendly messages instead of a huge traceback
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            print("\nERROR: free-tier quota reached for this model. Wait, or switch model, e.g.\n"
                  '   $env:GEMINI_MODEL="gemini-3-flash-preview"   (PowerShell)\n'
                  "Files of the steps that finished are already saved in ./" + OUT_DIR)
            sys.exit(1)
        if "404" in msg or "NOT_FOUND" in msg:
            print("\nERROR: model not found. Set another model name, e.g.\n"
                  '   $env:GEMINI_MODEL="gemini-3-flash-preview"   (PowerShell)')
            sys.exit(1)
        raise
    time.sleep(DELAY)
    return text


# ==============================================================
# Utilities
# ==============================================================
def words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def coverage(summary: str, terms) -> int:
    s = summary.lower()
    return sum(t.lower() in s for t in terms)


def save(name: str, content: str) -> None:
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def show(title: str, body: str) -> None:
    print(f"\n{title}")
    print("-" * 70)
    print(body)


# ==============================================================
# THE PIPELINE
# ==============================================================
def run_chain(llm, text: str) -> dict:
    r = {}

    print("\n[STEP 1/5] EXTRACT key points   (input: original text)")
    r["key_points"] = call_llm(llm, P1_EXTRACT.format(text=text))
    save("step1_key_points.txt", r["key_points"])
    show("Output of Step 1  ->  goes into Step 2", r["key_points"])

    print("\n[STEP 2/5] DRAFT summary        (input: key points)")
    r["draft"] = call_llm(llm, P2_DRAFT.format(key_points=r["key_points"], audience=AUDIENCE))
    save("step2_draft.txt", r["draft"])
    show("Output of Step 2  ->  goes into Step 3 and Step 4", r["draft"])

    print("\n[STEP 3/5] CRITIQUE the draft   (input: original text + draft)")
    r["critique"] = call_llm(llm, P3_CRITIQUE.format(text=text, draft=r["draft"]))
    save("step3_critique.txt", r["critique"])
    show("Output of Step 3  ->  goes into Step 4", r["critique"])

    print("\n[STEP 4/5] REFINE the summary   (input: text + draft + critique)")
    r["refined"] = call_llm(llm, P4_REFINE.format(text=text, draft=r["draft"], feedback=r["critique"],
                                                  max_words=MAX_WORDS, audience=AUDIENCE))
    save("step4_refined.txt", r["refined"])
    show("Output of Step 4  ->  goes into Step 5", r["refined"])

    print("\n[STEP 5/5] FORMAT the result    (input: refined summary)")
    r["final"] = call_llm(llm, P5_FORMAT.format(summary=r["refined"]))
    save("step5_final.txt", r["final"])
    show("FINAL OUTPUT OF THE CHAIN", r["final"])
    return r


def run_baseline(llm, text: str) -> str:
    print("\n[BASELINE] ONE single prompt (no chain)")
    out = call_llm(llm, P_BASELINE.format(text=text, max_words=MAX_WORDS, audience=AUDIENCE))
    save("baseline_single_prompt.txt", out)
    show("Baseline summary", out)
    return out


def compare(text, chain, baseline, use_terms) -> str:
    rows = [("Original text", words(text), None),
            ("Baseline (1 prompt)", words(baseline), coverage(baseline, KEY_TERMS) if use_terms else None),
            ("Chain refined summary (step 4)", words(chain["refined"]),
             coverage(chain["refined"], KEY_TERMS) if use_terms else None)]
    lines = ["Method                            Words   Key terms kept",
             "-" * 56]
    for name, w, c in rows:
        kept = "-" if c is None else f"{c}/{len(KEY_TERMS)}"
        lines.append(f"{name:<34}{w:<8}{kept}")
    lines.append("")
    lines.append(f"Compression (original -> chain summary): {words(text) / max(words(chain['refined']), 1):.1f}x")
    if not use_terms:
        lines.append("(Key-term check is only available for the built-in sample text.)")
    return "\n".join(lines)


def write_report(text, chain, baseline, table):
    md = [f"# Prompt Chaining - Summarization Report\n",
          f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}  |  Word limit: {MAX_WORDS}  |  Audience: {AUDIENCE}\n",
          "## Original text\n", text + "\n",
          "## Step 1 - Key points\n", chain["key_points"] + "\n",
          "## Step 2 - Draft summary\n", chain["draft"] + "\n",
          "## Step 3 - Editor critique\n", chain["critique"] + "\n",
          "## Step 4 - Refined summary\n", chain["refined"] + "\n",
          "## Step 5 - Final formatted output\n", chain["final"] + "\n",
          "## Baseline (single prompt)\n", baseline + "\n",
          "## Comparison\n", "```\n" + table + "\n```\n"]
    save("pipeline_report.md", "\n".join(md))


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(INPUT_FILE):
        text, use_terms = open(INPUT_FILE, encoding="utf-8").read().strip(), False
        print(f"Using your text from {INPUT_FILE} ({words(text)} words)")
    else:
        text, use_terms = SAMPLE_TEXT, True
        print(f"Using the built-in sample text ({words(text)} words)")
    llm = get_llm()

    chain = run_chain(llm, text)
    baseline = run_baseline(llm, text)

    table = compare(text, chain, baseline, use_terms)
    show("COMPARISON: single prompt vs prompt chain", table)
    write_report(text, chain, baseline, table)
    save("comparison.txt", table)

    print("\n" + "=" * 70)
    print(f"DONE. Files saved in ./{OUT_DIR}/ :")
    for n in sorted(os.listdir(OUT_DIR)):
        print("   -", n)
