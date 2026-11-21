"""Build the chat prompt for a question and read the SQL back out of the answer."""

import re
from typing import Any

SYSTEM = (
    "You are an expert in SQLite. Given a database schema and a question, write one SQLite "
    "query that answers the question. Reply with the SQL query only."
)
_FENCE = re.compile(r"```(?:\w+[ \t]*\n)?(.*?)(?:```|$)", re.DOTALL)
_QUERY_START = re.compile(r"\bselect\b|\bwith\s+\w+\s+as\s*\(", re.IGNORECASE)
_QUOTES = "'\"`"


def build_messages(schema: str, question: str) -> list[dict[str, str]]:
    """Return the chat messages that ask the model to answer the question with SQL."""
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def chat_prompt(tokenizer: Any, schema: str, question: str) -> str:  # noqa: ANN401
    """Return the full prompt text, ending where the model's answer begins."""
    messages = build_messages(schema, question)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _from_query_start(text: str) -> str:
    """Drop any words before the query, keeping brackets that open it."""
    match = _QUERY_START.search(text)
    if not match:
        return text
    start = match.start()
    while start > 0 and text[start - 1] in "( \t\n":
        start -= 1
    return text[start:]


def _first_statement(sql: str) -> str:
    """Cut at the first semicolon that is not inside a quoted string."""
    quote = ""
    for position, char in enumerate(sql):
        if quote:
            if char == quote:
                quote = ""
        elif char in _QUOTES:
            quote = char
        elif char == ";":
            return sql[:position]
    return sql


def extract_sql(answer: str) -> str:
    """Return the one SQL query in a model answer, without code fence, prose or a second query."""
    fenced = _FENCE.search(answer)
    text = fenced.group(1) if fenced else answer
    return " ".join(_first_statement(_from_query_start(text)).split())
