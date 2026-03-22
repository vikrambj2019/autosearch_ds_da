# Dexter — Your AI Data Scientist and Analyst

Hand Dexter a spreadsheet and a question. It answers you — and then tells you what you didn't think to ask.

No SQL. No dashboards. No waiting for an analyst.

---

## What Dexter Does

Ask Dexter a question in plain English. It runs the analysis, writes a clear report, and automatically goes one level deeper.

**You ask:** *"Did costs go up this month?"*
**Dexter answers that — and surfaces:** which member groups drove the increase, which counties contributed the most, and who the top movers were.

Think of it as a data scientist who doesn't just answer the question on the slide — they come back with the three follow-up slides you didn't know you needed.

**Ask Dexter things like:**
- *"What is the breakdown of costs by member status?"*
- *"Is there a significant difference between male and female members?"*
- *"How have costs trended from July to September?"*
- *"Compare this month to last month — what changed and why?"*
- *"Build me a model to predict which members will be high-cost next year"*

---

## Two Things We Did Differently

### 1. We gave the AI working code, not a blank page

Most data science work is procedural — 80% of it follows the same patterns every time. The real insight lives in the last 20%, where you dig into the right question in the right way.

We wrote that 80% ourselves: a set of tested, reliable Python functions — one for distributions, one for trends, one for group comparisons, one for period-over-period change. When Dexter runs an analysis, it starts from code that already works. The AI applies it intelligently to your question — it doesn't invent analysis from scratch.

This was the breakthrough. Earlier versions asked the AI to write analysis code from a blank slate. It kept breaking. Handing it working code as a starting point changed everything.

### 2. We profile your data before we touch your question

Before Dexter reads your question, it reads your data. It checks what columns exist, what they contain, flags any data quality issues, and figures out how the data is structured — one row per person, or one row per person per month?

Only then does it match your question to the right columns. This means Dexter understands what you're asking — "gender" maps to the right column even if the file calls it `SEX`. And when something breaks, we know exactly which column, which step, and why — not a cryptic error halfway through.

---

## The Self-Improving Model Builder

Ask Dexter to build a predictive model and it doesn't stop at one pass. It runs multiple rounds of improvement automatically — inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

- **Round 1** — The AI reads your data schema and writes a baseline classification model pipeline from scratch.
- **Rounds 2–6** — The AI sees the best pipeline so far plus a log of every previous experiment. It makes exactly **one targeted change** — for example, switching from Random Forest to LightGBM, or adding feature selection. Dexter runs the new pipeline, scores it, and decides: keep the change if it improved, discard it if it did not.
- **Stops automatically** when the model hits a quality threshold, stops improving for two rounds in a row, or reaches 6 rounds.

Each model is scored on five dimensions: prediction accuracy (AUC), precision-recall balance (F1), training speed, how well the top features explain the model (SHAP coverage), and how interpretable the results are to a non-technical audience.

The AI proposes changes but **never decides what to keep** — that is always Dexter's job. This means experiments are consistent, reproducible, and the log clearly shows what each round changed and whether it helped.

---

## Other Clever Things Under the Hood

### The AI never gets the direction wrong

Every comparison result — "Group A is higher than Group B by 23%" — is computed by Python before the AI ever sees it. The AI is told to copy that sentence verbatim, not re-derive it. In earlier versions, the AI would sometimes report the wrong group as higher. This eliminates that entirely.

### High-value segments never get buried

When a column has dozens of categories (like County with 50+ values), most systems just show "top 10 by count" and lump the rest into "Other." Dexter uses a smarter rule: keep whichever categories cover 95% of total spend, or top 10 by count — whichever shows more. A county with 50 members but 30% of total cost will never disappear into "Other."

### Panel data is handled correctly

When the same person appears across multiple months, averaging their monthly averages gives you the wrong number. Dexter always divides total sum by total observations — the statistically correct approach — not the common shortcut that quietly introduces error.

### If the AI hallucinates a column name, Dexter catches it

When Dexter matches your question to columns in your data, the AI sometimes returns a column name that doesn't exist. Dexter validates every column name before using it and falls back to a sensible default if the AI got it wrong. The analysis continues — it never crashes silently with bad data.

---

## Guardrails

- Python computes every number in the report — the AI only writes the narrative
- Your data stays on your machine
- Every report saves with a timestamp for auditability
- A safety layer blocks the AI from deleting files, making network calls, or running dangerous system commands

---

## Built With

- [Claude Agent SDK](https://www.anthropic.com) — AI reasoning and report writing
- Python — all statistical computation (pandas, scipy, scikit-learn, LightGBM, XGBoost)
- Runs entirely locally — your data never leaves your machine
