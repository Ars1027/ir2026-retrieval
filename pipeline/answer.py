"""Answer strategies that only use the local evidence.

Two explicit modes:

``extractive``
    Deterministic span selection from the evidence quotes. No model, tokens 0.
    The answer is a short verbatim sentence, never the whole passage.

``model``
    A fixed question-answering call over the same quotes. The model may only
    answer from those quotes; failures return a reason and the pipeline keeps
    the retrieved documents and evidence.

Neither mode invents text: an answer that cannot be supported is reported as
``no_answer``.
"""
from __future__ import annotations

import re

from pipeline.model_client import ModelClient, ModelClientError
from retrieval.bm25 import tokenize

SENTENCE_RE = re.compile(r"[^.!?\n]{20,400}[.!?]?")
NUMBER_RE = re.compile(r"\d[\d.,]*")
NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = ("You answer a research question using ONLY the numbered evidence passages. "
                 "Reply with a short answer phrase (max 12 words). "
                 f"If the passages do not contain the answer, reply exactly {NO_ANSWER}. "
                 "Do not explain, do not cite, do not use outside knowledge.")


def _question_terms(question: str) -> list:
    seen = []
    for token in tokenize(question or ""):
        if token not in seen:
            seen.append(token)
    return seen


def split_sentences(text: str) -> list:
    sentences = [m.group(0).strip() for m in SENTENCE_RE.finditer(text or "")]
    return [s for s in sentences if len(s) >= 20]


def _score_sentence(sentence: str, terms: list, numbers: set):
    """Score a candidate span; ``None`` rejects lists, tables and number dumps."""
    if sentence.count(",") > 8:
        return None
    digits = sum(ch.isdigit() for ch in sentence)
    if len(sentence) and digits / len(sentence) > 0.25:
        return None
    lowered = sentence.lower()
    matched = sum(1 for term in terms if term in lowered)
    numeric = sum(1 for number in numbers if number and number in sentence)
    return matched + 3 * numeric, matched, numeric, -len(sentence)


def extractive_answer(question: str, quotes: list, max_chars: int = 200) -> tuple:
    """Return ``(answer, note)``; empty answer means the evidence is insufficient."""
    terms = _question_terms(question)
    numbers = {value.rstrip(".,") for value in NUMBER_RE.findall(question or "")}
    if not terms:
        return "", "empty question"
    best = None
    for quote in quotes:
        for sentence in split_sentences(quote["text"]):
            key = _score_sentence(sentence, terms, numbers)
            if key is None:
                continue
            if best is None or key > best[0]:
                best = (key, sentence)
    if best is None:
        return "", "no complete sentence in evidence"
    _, matched, numeric, _ = best[0]
    sentence = best[1]
    needed = 1 if len(terms) <= 2 else 2
    if matched < needed and numeric == 0:
        return "", "evidence shares too few terms with the question"
    answer = sentence.strip().strip('"“”')
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0].strip()
    return answer, "extractive span from the top evidence quote"


def model_answer(question: str, quotes: list, settings: dict) -> tuple:
    """Return ``(answer, usage, note)``; raises ModelClientError when unavailable."""
    client = ModelClient(settings)
    config = settings.get("answer") or {}
    max_chars = int(config.get("max_answer_chars") or 200)
    if not quotes:
        return "", {"model_input_tokens": 0, "model_output_tokens": 0,
                    "token_notes": "No evidence quotes; model was not called."}, "no evidence"
    blocks = []
    for index, quote in enumerate(quotes, 1):
        blocks.append(f"[{index}] document {quote['doc_id']}\n{quote['text'][:800]}")
    user = f"Question: {question}\n\nEvidence passages:\n" + "\n\n".join(blocks)
    text, usage = client.complete(SYSTEM_PROMPT, user)
    if not text or text.strip().upper().startswith(NO_ANSWER):
        usage["token_notes"] = (usage.get("token_notes") or "") + " Model answered NO_ANSWER."
        return "", usage, "model reported no answer in the evidence"
    answer = text.splitlines()[0].strip().strip('"“”')
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0].strip()
    return answer, usage, f"model answer from {client.model}"


def make_answer(question: str, quotes: list, settings: dict) -> dict:
    """Dispatch on ``answer.mode``; always returns a dict, never raises."""
    config = settings.get("answer") or {}
    mode = config.get("mode") or "extractive"
    if mode == "off":
        return {"answer": "", "note": "answer generation disabled", "error": None,
                "usage": {"model_input_tokens": 0, "model_output_tokens": 0,
                          "token_notes": "No model used (answer.mode=off)."}}
    if mode == "model":
        try:
            answer, usage, note = model_answer(question, quotes, settings)
            return {"answer": answer, "note": note, "error": None, "usage": usage}
        except ModelClientError as exc:
            return {"answer": "", "note": "model unavailable; kept retrieval and evidence",
                    "error": str(exc),
                    "usage": {"model_input_tokens": None, "model_output_tokens": None,
                              "token_notes": f"Model call failed: {exc}"}}
    answer, note = extractive_answer(question, quotes,
                                     int(config.get("max_answer_chars") or 200))
    return {"answer": answer, "note": note, "error": None,
            "usage": {"model_input_tokens": 0, "model_output_tokens": 0,
                      "token_notes": "No model used (extractive answer strategy)."}}
