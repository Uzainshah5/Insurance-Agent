"""
Insurellm Answer module.
Re-exports answer_question and fetch_context from answer_langchain.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from implementation.answer_langchain import *

if __name__ == "__main__":
    answer, used_chunks = answer_question("Who is the CEO of Insurellm?")
    print("\nANSWER:\n", answer)
    print("\nNumber of chunks used:", len(used_chunks))

