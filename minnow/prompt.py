"""Build the chat prompt for a question and read the SQL back out of the answer."""

import re

SYSTEM = (
    "You are an expert in SQLite. Given a database schema and a question, write one SQLite "
    "query that answers the question. Reply with the SQL query only."
)
_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


def build_messages(schema: str, question: str) -> list[dict[str, str]]:
    """Return the chat messages that ask the model to answer the question with SQL."""
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_sql(answer: str) -> str:
    """Return the SQL query from a model answer, with any code fence and trailing text removed."""
    fenced = _FENCE.search(answer)
    sql = fenced.group(1) if fenced else answer
    sql = sql.strip()
    end = sql.find(";")
    if end != -1:
        sql = sql[:end]
    return " ".join(sql.split())
