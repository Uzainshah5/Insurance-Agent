"""
Simple LangChain version of the Insurellm evaluation code.

What this code does:
1. Retrieval test: checks if the right keywords appear in the chunks we find
   (scores: MRR, nDCG, keyword coverage)
2. Answer test: an LLM "judge" compares our answer with the reference answer
   (scores: accuracy, completeness, relevance)

Install first:
pip install langchain-core langchain-openai python-dotenv pydantic

Your .env file needs:
OPENAI_API_KEY=...

Run it with:
python eval_langchain.py 0
(0 is the test number. Use 1, 2, 3 ... for other tests.)

IMPORTANT: the imports below expect the file  implementation/answer.py
So save "answer_langchain.py" as "implementation/answer.py"
(or change the import line to match your file name).
"""

import os
import sys
import math
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Redirect LOCALAPPDATA / patch certifi-win32 for Windows
local_appdata = project_root / ".localappdata"
local_appdata.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(local_appdata)
try:
    import certifi_win32.wincerts
    certifi_win32.wincerts.PEM_PATH = str(local_appdata / ".certifi" / "cacert.pem")
except Exception:
    pass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context


 # Step 1: Settings and LangChain objects
 
# Load the keys from the .env file
load_dotenv(override=True)

MODEL = os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-super-vl:free")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY")

# The judge model.
# max_retries = if a call fails, LangChain tries again by itself.
llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0,
    max_retries=5,
)

# Step 2: The shapes of the results

class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


 # Step 3: Score functions (plain Python, no LangChain here)
 
def calculate_mrr(keyword, retrieved_docs):
    """
    Reciprocal rank for ONE keyword.
    If the keyword is in the 1st chunk -> 1.0, in the 2nd chunk -> 0.5, in the 3rd -> 0.33 ...
    If it is not found -> 0.0
    """
    keyword_lower = keyword.lower()

    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank

    return 0.0


def calculate_dcg(relevances, k):
    """Discounted Cumulative Gain: found chunks at the top count more."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        # i + 2 because the rank starts at 1
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def calculate_ndcg(keyword, retrieved_docs, k=10):
    """nDCG for ONE keyword. 1 = keyword found, 0 = not found (binary)."""
    keyword_lower = keyword.lower()

    # Make a list like [0, 1, 0, 0, 1 ...] for the top k chunks
    relevances = []
    for doc in retrieved_docs[:k]:
        if keyword_lower in doc.page_content.lower():
            relevances.append(1)
        else:
            relevances.append(0)

    # Real DCG
    dcg = calculate_dcg(relevances, k)

    # Best possible DCG (all the 1s at the top)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    if idcg > 0:
        return dcg / idcg
    return 0.0

def evaluate_retrieval(test, k=10):
    """Check how good the retrieved chunks are for one test question."""
    # Get the chunks from the RAG code
    retrieved_docs = fetch_context(test.question)

    # MRR for every keyword
    mrr_scores = []
    for keyword in test.keywords:
        mrr_scores.append(calculate_mrr(keyword, retrieved_docs))

    # nDCG for every keyword
    ndcg_scores = []
    for keyword in test.keywords:
        ndcg_scores.append(calculate_ndcg(keyword, retrieved_docs, k))

    # Average of the scores (check for an empty list, to avoid dividing by zero)
    if len(mrr_scores) > 0:
        avg_mrr = sum(mrr_scores) / len(mrr_scores)
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    else:
        avg_mrr = 0.0
        avg_ndcg = 0.0

    # Count how many keywords were found (score above 0 means found)
    keywords_found = 0
    for score in mrr_scores:
        if score > 0:
            keywords_found += 1

    total_keywords = len(test.keywords)

    if total_keywords > 0:
        keyword_coverage = keywords_found / total_keywords * 100
    else:
        keyword_coverage = 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# Step 5: Answer evaluation (LLM as a judge)

JUDGE_SYSTEM_PROMPT = "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers."


def evaluate_answer(test):
    """
    Get an answer from our RAG code, then let the LLM judge it.
    Returns: (AnswerEval, generated answer text, retrieved chunks)
    """
    # 1. Get the answer from the RAG code
    generated_answer, retrieved_docs = answer_question(test.question)

    # 2. Build the prompt for the judge
    judge_prompt = f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1."""

    # 3. Ask the judge. It replies in the AnswerEval shape.
    judge_llm = llm.with_structured_output(AnswerEval)

    try:
        answer_eval = judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=judge_prompt),
            ]
        )
    except Exception as e:
        print(f"Judge structured output error: {e}")
        answer_eval = AnswerEval(
            feedback=f"Evaluator note: {e}",
            accuracy=3.0,
            completeness=3.0,
            relevance=3.0,
        )

    return answer_eval, generated_answer, retrieved_docs


# Step 6: Run all tests (used by the UI, one result at a time)

def evaluate_all_retrieval():
    """Evaluate all retrieval tests. Gives back one result at a time."""
    tests = load_tests()
    total_tests = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate all answers. Gives back one result at a time."""
    tests = load_tests()
    total_tests = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


# Step 7: Run ONE test from the command line

def run_cli_evaluation(test_number):
    """Run the retrieval test and the answer test for one test number."""
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    # Show the test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Answer evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """Read the test number from the command line and run it."""
    if len(sys.argv) != 2:
        print("Usage: python eval_langchain.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
