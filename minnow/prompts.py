"""The schema text shown to the model for each question, built once per database."""

from collections.abc import Iterable
from pathlib import Path

from minnow.spider import SpiderExample, database_path, schemas
from minnow.values import hint, mentioned, value_index


class PromptBook:
    """Hold every database's schema text, and optionally its values for question hints."""

    def __init__(
        self,
        spider: Path,
        examples: Iterable[SpiderExample],
        sample_rows: int = 0,
        value_hints: bool = False,
    ) -> None:
        """Read the schema (and, with value_hints, the text values) of every database used."""
        examples = list(examples)
        self._schemas = schemas(spider, examples, sample_rows)
        self._values = {}
        if value_hints:
            self._values = {db: value_index(database_path(spider, db)) for db in self._schemas}

    def schema(self, example: SpiderExample) -> str:
        """Return the schema text for a question, with the values it mentions when enabled."""
        text = self._schemas[example.db_id]
        if example.db_id not in self._values:
            return text
        found = hint(mentioned(self._values[example.db_id], example.question))
        return f"{text}\n\n{found}" if found else text
