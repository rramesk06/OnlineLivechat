"""
PDF RAG + Graph RAG - DeepEval Evaluation

This evaluator:
1. Checks FastAPI health
2. Finds the indexed PDF
3. Loads evaluation test cases
4. Calls the RAG chat endpoint
5. Calls the hybrid retrieval endpoint
6. Builds DeepEval LLMTestCase objects
7. Runs:
   - Answer Relevancy
   - Faithfulness
   - Contextual Relevancy
   - Contextual Precision
   - Contextual Recall

Windows stability:
- DeepEval cache is disabled
- DeepEval async/parallel execution is disabled
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from deepeval import evaluate
from deepeval.evaluate import CacheConfig, AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

TEST_CASES_FILE = BASE_DIR / "evaluation" / "test_cases.json"

REQUEST_TIMEOUT = 120

TOP_K = 5


# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def print_error(message: str) -> None:
    print()
    print(f"ERROR: {message}")
    print()


# ============================================================
# HTTP HELPERS
# ============================================================

def api_get(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    url = f"{API_URL}{endpoint}"

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"GET {endpoint} failed "
            f"({response.status_code}): "
            f"{response.text}"
        )

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"GET {endpoint} returned invalid JSON: "
            f"{response.text}"
        ) from exc


def api_post(
    endpoint: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    url = f"{API_URL}{endpoint}"

    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        raise RuntimeError(
            f"POST {endpoint} failed "
            f"({response.status_code}): "
            f"{response.text}"
        )

    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"POST {endpoint} returned invalid JSON: "
            f"{response.text}"
        ) from exc


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

def check_api() -> None:

    try:
        result = api_get("/health")
    except Exception as exc:
        raise RuntimeError(
            f"FastAPI is not available at {API_URL}. "
            f"Start FastAPI first.\n{exc}"
        ) from exc

    print("✓ FastAPI is running")


# ============================================================
# DOCUMENT DISCOVERY
# ============================================================

def get_documents() -> List[Dict[str, Any]]:

    result = api_get("/documents")

    if isinstance(result, list):
        return result

    if isinstance(result, dict):

        documents = result.get("documents")

        if isinstance(documents, list):
            return documents

        data = result.get("data")

        if isinstance(data, list):
            return data

    raise RuntimeError(
        "Unexpected /documents response:\n"
        f"{json.dumps(result, indent=2, default=str)}"
    )


def get_document_id(document: Dict[str, Any]) -> str:

    possible_keys = [
        "document_id",
        "id",
        "documentId",
    ]

    for key in possible_keys:

        value = document.get(key)

        if value:
            return str(value)

    raise RuntimeError(
        "Could not find document ID in document response:\n"
        f"{json.dumps(document, indent=2, default=str)}"
    )


def get_document_name(document: Dict[str, Any]) -> str:

    possible_keys = [
        "filename",
        "file_name",
        "name",
        "original_filename",
    ]

    for key in possible_keys:

        value = document.get(key)

        if value:
            return str(value)

    return "Unknown document"


# ============================================================
# LOAD TEST CASES
# ============================================================

def load_test_cases() -> List[Dict[str, Any]]:

    if not TEST_CASES_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation test case file not found:\n"
            f"{TEST_CASES_FILE}"
        )

    with TEST_CASES_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "test_cases",
            "tests",
            "questions",
        ]:

            value = data.get(key)

            if isinstance(value, list):
                return value

    raise RuntimeError(
        "Invalid test_cases.json format."
    )


def get_question(test_case: Dict[str, Any]) -> str:

    for key in [
        "question",
        "input",
        "query",
    ]:

        value = test_case.get(key)

        if value:
            return str(value)

    raise RuntimeError(
        f"Test case does not contain a question:\n"
        f"{json.dumps(test_case, indent=2)}"
    )


def get_expected_output(
    test_case: Dict[str, Any],
) -> str:

    for key in [
        "expected_output",
        "expected",
        "answer",
        "ground_truth",
    ]:

        value = test_case.get(key)

        if value is not None:

            if isinstance(value, str):
                return value

            return json.dumps(
                value,
                ensure_ascii=False,
            )

    # DeepEval can still evaluate relevancy/faithfulness
    # without a reference answer.
    return ""


# ============================================================
# CHAT / RAG ANSWER
# ============================================================

def get_chat_answer(
    document_id: str,
    question: str,
) -> Dict[str, Any]:

    payload = {
        "document_id": document_id,
        "question": question,
    }

    return api_post(
        f"/documents/{document_id}/chat",
        payload,
    )


def extract_answer(
    result: Dict[str, Any],
) -> str:

    possible_keys = [
        "answer",
        "response",
        "message",
        "result",
        "content",
        "output",
    ]

    for key in possible_keys:

        value = result.get(key)

        if value is None:
            continue

        if isinstance(value, str):
            if value.strip():
                return value.strip()

        if isinstance(value, dict):

            nested = extract_answer(value)

            if nested:
                return nested

    # Some APIs return data.message/data.answer
    data = result.get("data")

    if isinstance(data, dict):

        nested = extract_answer(data)

        if nested:
            return nested

    return json.dumps(
        result,
        ensure_ascii=False,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def get_retrieval(
    document_id: str,
    question: str,
    top_k: int = TOP_K,
) -> Dict[str, Any]:

    """
    IMPORTANT:
    /retrieve requires all three:
      document_id
      question
      top_k
    """

    payload = {
        "document_id": document_id,
        "question": question,
        "top_k": top_k,
    }

    return api_post(
        "/retrieve",
        payload,
    )


# ============================================================
# RETRIEVAL RESULT EXTRACTION
# ============================================================

def extract_text_from_item(
    item: Any,
) -> str:

    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item)

    # Most likely keys first
    possible_keys = [
        "text",
        "content",
        "chunk",
        "document",
        "page_content",
        "passage",
        "description",
        "value",
    ]

    for key in possible_keys:

        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # LangChain-style nested document
    document = item.get("document")

    if isinstance(document, dict):

        nested = extract_text_from_item(document)

        if nested:
            return nested

    return json.dumps(
        item,
        ensure_ascii=False,
    )


def extract_result_list(
    result: Dict[str, Any],
    keys: List[str],
) -> List[Any]:

    for key in keys:

        value = result.get(key)

        if isinstance(value, list):
            return value

    return []


def build_retrieval_context(
    retrieval_result: Dict[str, Any],
) -> tuple[List[str], str]:

    """
    Prefer hybrid results because this evaluates
    the actual hybrid RAG retrieval.

    Fallback:
      hybrid -> vector -> graph
    """

    hybrid_results = extract_result_list(
        retrieval_result,
        [
            "hybrid_results",
            "hybrid",
            "hybrid_search",
            "results",
        ],
    )

    vector_results = extract_result_list(
        retrieval_result,
        [
            "vector_results",
            "vector",
            "vector_search",
        ],
    )

    graph_results = extract_result_list(
        retrieval_result,
        [
            "graph_results",
            "graph",
            "graph_search",
        ],
    )

    selected_results: List[Any] = []
    source = "none"

    if hybrid_results:

        selected_results = hybrid_results
        source = "hybrid"

    elif vector_results:

        selected_results = vector_results
        source = "vector"

    elif graph_results:

        selected_results = graph_results
        source = "graph"

    contexts: List[str] = []

    for item in selected_results:

        text = extract_text_from_item(item)

        if text and text.strip():

            contexts.append(text.strip())

    # Remove duplicate contexts while preserving order
    unique_contexts: List[str] = []
    seen = set()

    for context in contexts:

        normalized = context.strip()

        if normalized.lower() in seen:
            continue

        seen.add(normalized.lower())
        unique_contexts.append(normalized)

    return unique_contexts, source


# ============================================================
# METRICS
# ============================================================

def build_metrics():

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.5,
        model="gpt-5.4",
        include_reason=True,
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.5,
        model="gpt-5.4",
        include_reason=True,
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.5,
        model="gpt-5.4",
        include_reason=True,
    )

    contextual_precision = ContextualPrecisionMetric(
        threshold=0.5,
        model="gpt-5.4",
        include_reason=True,
    )

    contextual_recall = ContextualRecallMetric(
        threshold=0.5,
        model="gpt-5.4",
        include_reason=True,
    )

    return [
        answer_relevancy,
        faithfulness,
        contextual_relevancy,
        contextual_precision,
        contextual_recall,
    ]


# ============================================================
# PREPARE DEEPEVAL CASES
# ============================================================

def prepare_test_cases(
    document_id: str,
    evaluation_cases: List[Dict[str, Any]],
) -> List[LLMTestCase]:

    deepeval_cases: List[LLMTestCase] = []

    print_header(
        f"Preparing {len(evaluation_cases)} evaluation test cases"
    )

    for index, evaluation_case in enumerate(
        evaluation_cases,
        start=1,
    ):

        question = get_question(evaluation_case)

        print(
            f"[{index}/{len(evaluation_cases)}] "
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Generate RAG answer
        # ----------------------------------------------------

        chat_result = get_chat_answer(
            document_id=document_id,
            question=question,
        )

        actual_output = extract_answer(
            chat_result
        )

        # ----------------------------------------------------
        # Get retrieval evidence
        # ----------------------------------------------------

        retrieval_result = get_retrieval(
            document_id=document_id,
            question=question,
            top_k=TOP_K,
        )

        retrieval_context, retrieval_source = (
            build_retrieval_context(
                retrieval_result
            )
        )

        print(
            f"    Source: {retrieval_source}"
        )

        print(
            f"    Retrieved passages: "
            f"{len(retrieval_context)}"
        )

        if retrieval_context:

            preview = retrieval_context[0]

            preview = preview.replace(
                "\n",
                " ",
            )

            if len(preview) > 150:
                preview = preview[:150] + "..."

            print(
                f"    First passage: {preview}"
            )

        else:

            print(
                "    WARNING: No retrieval context found"
            )

        expected_output = get_expected_output(
            evaluation_case
        )

        # ----------------------------------------------------
        # DeepEval test case
        # ----------------------------------------------------

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        deepeval_cases.append(test_case)

        print()

    return deepeval_cases


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "PDF RAG + Graph RAG - DeepEval Evaluation"
    )

    # --------------------------------------------------------
    # 1. Check API
    # --------------------------------------------------------

    try:

        check_api()

    except Exception as exc:

        print_error(str(exc))
        sys.exit(1)

    print()

    # --------------------------------------------------------
    # 2. Find document
    # --------------------------------------------------------

    try:

        documents = get_documents()

    except Exception as exc:

        print_error(
            f"Could not load documents.\n{exc}"
        )
        sys.exit(1)

    if not documents:

        print_error(
            "No indexed documents found."
        )
        sys.exit(1)

    # Prefer Spotify architecture PDF if available
    selected_document = None

    for document in documents:

        filename = get_document_name(
            document
        ).lower()

        if (
            "spotify" in filename
            and filename.endswith(".pdf")
        ):

            selected_document = document
            break

    if selected_document is None:

        selected_document = documents[0]

    document_id = get_document_id(
        selected_document
    )

    document_name = get_document_name(
        selected_document
    )

    print(
        f"✓ Evaluation document: "
        f"{document_name}"
    )

    print(
        f"✓ Document ID: "
        f"{document_id}"
    )

    # --------------------------------------------------------
    # 3. Load evaluation cases
    # --------------------------------------------------------

    try:

        evaluation_cases = load_test_cases()

    except Exception as exc:

        print_error(
            f"Could not load evaluation cases.\n{exc}"
        )
        sys.exit(1)

    print(
        f"\n✓ Loaded "
        f"{len(evaluation_cases)} "
        f"evaluation test cases"
    )

    # --------------------------------------------------------
    # 4. Prepare DeepEval test cases
    # --------------------------------------------------------

    try:

        test_cases = prepare_test_cases(
            document_id=document_id,
            evaluation_cases=evaluation_cases,
        )

    except Exception as exc:

        print_error(
            f"Could not prepare DeepEval test cases.\n{exc}"
        )
        sys.exit(1)

    print_header(
        f"Prepared {len(test_cases)} DeepEval test cases"
    )

    if not test_cases:

        print_error(
            "No DeepEval test cases were prepared."
        )
        sys.exit(1)

    # --------------------------------------------------------
    # 5. Metrics
    # --------------------------------------------------------

    metrics = build_metrics()

    # --------------------------------------------------------
    # 6. DeepEval
    #
    # IMPORTANT WINDOWS FIX:
    #
    # use_cache=False
    # write_cache=False
    #
    # prevents the NoneType cache crash.
    #
    # run_async=False
    #
    # prevents DeepEval parallel cache interaction.
    # --------------------------------------------------------

    print_header(
        "Running DeepEval..."
    )

    try:

        evaluate(
            test_cases=test_cases,
            metrics=metrics,
            cache_config=CacheConfig(
                use_cache=False,
                write_cache=False,
            ),
            async_config=AsyncConfig(
                run_async=False,
            ),
        )

    except Exception as exc:

        print_error(
            "DeepEval execution failed."
        )

        print(
            "Exception:"
        )

        print(
            repr(exc)
        )

        print()

        print(
            "The RAG retrieval preparation completed "
            "successfully. The failure occurred during "
            "DeepEval metric execution."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # 7. Done
    # --------------------------------------------------------

    print_header(
        "DeepEval evaluation completed successfully"
    )

    print(
        "✓ RAG answers generated"
    )

    print(
        "✓ Hybrid retrieval contexts collected"
    )

    print(
        "✓ DeepEval metrics executed"
    )

    print(
        "✓ Windows cache/parallel execution disabled"
    )


if __name__ == "__main__":
    main()