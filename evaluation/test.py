"""
Test questions loader.

This file does NOT use an LLM, so it does not need LangChain.
It only reads the test questions from the file "tests.jsonl".
(JSONL means: one JSON object on each line.)

Save this file as:  evaluation/test.py
The file "tests.jsonl" must be in the same folder.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field


# The path of tests.jsonl (it is in the same folder as this file)
TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """One test question with the keywords and the reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests():
    """Read all test questions from the JSONL file and return them as a list."""
    tests = []

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines (they would crash json.loads)
            if line == "":
                continue

            # Turn the text line into a Python dictionary
            data = json.loads(line)

            # Turn the dictionary into a TestQuestion object
            tests.append(TestQuestion(**data))

    return tests
