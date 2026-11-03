"""Answer strategies that only use the local evidence.

Two explicit modes:

``extractive``
    Deterministic span selection. No model, tokens 0. A short answer span is
    chosen from the retrieved documents by matching the question's expected
    answer type (title, person, organisation, quantity, date) and the question
    keywords around it; the sentence holding the span becomes the evidence
    quote, so every answer stays verbatim checkable. When no span is confident
    enough the strategy returns an empty answer and the record becomes
    ``no_answer``.

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

# --- answer typing cues -----------------------------------------------------

STOPWORDS = frozenset("""a an the of in on at for to by with from and or as is are was were be been
being that this these those it its their his her they them then than which who whom whose what when
where why how do does did done have has had not no nor but if while about into over under between
after before during such other another also more most less least very can could would should may
might must shall will""".split())

PERSON_NOUNS = ("author", "writer", "person", "husband", "wife", "celebrity", "controller",
                "director", "actor", "actress", "founder", "ceo", "president", "singer",
                "artist", "individual", "researcher", "scientist", "player", "producer",
                "composer", "painter", "poet", "politician", "man", "woman", "boy", "girl",
                "child", "king", "queen", "narrator", "editor", "illustrator", "designer")
WORK_NOUNS = ("book", "novel", "film", "movie", "series", "show", "song", "album", "story",
              "paper", "article", "podcast", "game", "publication", "work", "episode", "track",
              "play", "poem", "tale", "title", "manuscript", "report", "essay", "post")
ORG_NOUNS = ("company", "firm", "corporation", "organisation", "organization", "brand",
             "manufacturer", "publisher", "agency", "studio", "label", "airline", "bank",
             "university", "institute", "team", "club", "chain", "retailer", "startup")

QUANTITY_RE_Q = re.compile(r"\bhow many\b|\bhow much\b|\bwhat percentage\b|\bpercentage of\b|"
                           r"\bnumber of\b|\bamount of\b|\bhow old\b|\bwhat proportion\b|"
                           r"\bwhat share\b", re.I)
PERSON_RE_Q = re.compile(r"\b(?:full |first and last )?name of the (?:%s)\b|"
                         r"\bwho (?:was|is|were|are|wrote|created|founded|directed|produced|"
                         r"composed|painted|played|invented|discovered)\b|\bwhich (?:author|person|"
                         r"actor|actress|singer|scientist)\b" % "|".join(PERSON_NOUNS), re.I)
WORK_RE_Q = re.compile(r"\btitle of\b|\bname of the (?:%s)\b|\bwhat is the (?:title|name) of the "
                       r"(?:%s)\b|\bwhich (?:book|film|movie|series|song|album|paper|article|story)\b"
                       % ("|".join(WORK_NOUNS), "|".join(WORK_NOUNS)), re.I)
ORG_RE_Q = re.compile(r"\b(?:%s)(?:'s)? (?:full )?name\b|\bname of the (?:%s)\b|\bwhich (?:%s)\b"
                      % ("|".join(ORG_NOUNS), "|".join(ORG_NOUNS), "|".join(ORG_NOUNS)), re.I)
DATE_RE_Q = re.compile(r"\bwhat year\b|\bwhich year\b|\bin what year\b|\bwhat date\b|"
                       r"\bwhen (?:was|did|were|is)\b", re.I)

# --- span patterns ----------------------------------------------------------

QUOTED_RE = re.compile(r"[\"“”']([^\"“”'\n]{2,120})[\"“”']")
PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
MONEY_RE = re.compile(r"[$€£]\s?\d[\d.,]*\s?(?:billion|million|thousand|bn|m|k)?", re.I)
QUANTITY_RE = re.compile(r"\b\d+(?:[.,]\d+)*\s?(?:%|percent|million|billion|thousand|bn|m|k|"
                         r"kg|km|mi|metres?|meters?|feet|ft|cm|mm|hours?|minutes?|days?|years?|"
                         r"seconds?|people|pages?|species|units?)?\b", re.I)
DATE_RE = re.compile(r"\b(?:\d{1,2}\s+)?(?:january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)(?:\s+\d{1,2})?(?:,?\s+\d{4})?\b", re.I)

WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*\.?|\d+|[^\sA-Za-z\d]+")
TITLE_JOINERS = frozenset("of the and in on for to a an at by with from or nor but".split())
NAME_PARTICLES = frozenset("van von der de la del di da le du bin ibn al".split())


def _question_terms(question: str) -> list:
    seen = []
    for token in tokenize(question or ""):
        if token not in seen:
            seen.append(token)
    return seen


def _keyword_terms(question: str) -> list:
    """Content words used to locate the answering sentence."""
    terms = []
    for token in _question_terms(question):
        if len(token) < 4 or token in STOPWORDS or token.isdigit():
            continue
        if token not in terms:
            terms.append(token)
    return terms


def expected_type(question: str) -> str:
    """Coarse answer type inferred from the question wording."""
    text = question or ""
    if QUANTITY_RE_Q.search(text):
        return "quantity"
    if WORK_RE_Q.search(text):
        return "title"
    if re.search(r"\b(?:title|book|paper|film|movie|series|song|album|story|novel|article|podcast)\b",
                 text, re.I) and not ORG_RE_Q.search(text):
        return "title"
    if ORG_RE_Q.search(text):
        return "org"
    if PERSON_RE_Q.search(text) or re.search(r"\bname of the\b", text, re.I):
        return "person"
    if DATE_RE_Q.search(text):
        return "date"
    return "title"


def split_sentences(text: str) -> list:
    sentences = [m.group(0).strip() for m in SENTENCE_RE.finditer(text or "")]
    return [s for s in sentences if len(s) >= 20]


def _score_sentence(sentence: str, terms: list, numbers: set):
    """Score a candidate sentence; ``None`` rejects lists, tables and number dumps."""
    if sentence.count(",") > 8:
        return None
    digits = sum(ch.isdigit() for ch in sentence)
    if len(sentence) and digits / len(sentence) > 0.25:
        return None
    lowered = sentence.lower()
    matched = sum(1 for term in terms if term in lowered)
    numeric = sum(1 for number in numbers if number and number in sentence)
    return matched + 3 * numeric, matched, numeric, -len(sentence)


# --- candidate spans -------------------------------------------------------


def _clean_span(raw: str) -> str:
    span = raw.strip().strip('"“”\'')
    span = span.strip(" ,;:.-–—()[]")
    return span.strip()


def _is_capitalised(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Za-z'’\-.]*", token))


def _title_candidates(text: str) -> list:
    """Capitalised multi-word runs that look like titles or names."""
    out = []
    tokens = [(m.group(0), m.start(), m.end()) for m in WORD_TOKEN_RE.finditer(text or "")]
    run = []

    def flush():
        while run and not _is_capitalised(run[0][0]):
            run.pop(0)
        while run and not _is_capitalised(run[-1][0]):
            run.pop()
        if run:
            wordy = [item for item in run if re.search(r"[A-Za-z]", item[0])]
            if len(wordy) >= 2 and any(len(item[0]) > 2 for item in run):
                start, end = run[0][1], run[-1][2]
                span = _clean_span(text[start:end])
                if span:
                    out.append({"text": span, "start": start, "end": end})
        run.clear()

    for token, start, end in tokens:
        bare = token.strip(".,;:!?")
        if _is_capitalised(bare) or bare in TITLE_JOINERS or bare.lower() in NAME_PARTICLES:
            run.append((bare, start, end))
        else:
            flush()
    flush()
    return out


def _anchored_candidates(text: str, anchor: str, words_back: int = 12) -> list:
    """Sentence-case spans ending/starting with a hinted phrase (titles)."""
    out = []
    if not anchor:
        return out
    for match in re.finditer(re.escape(anchor), text or "", re.I):
        tokens = list(re.finditer(r"\S+", text[:match.start()]))
        if len(tokens) < 3:
            continue
        tail = tokens[-words_back:]
        start = tail[0].start()
        span = _clean_span(text[start:match.end()])
        span = re.sub(r"^(?:the|a|an|of|and|in|on|for|to|by|with)\s+", "", span, flags=re.I)
        start = text.find(span, start) if span else start
        if span and 6 <= len(span) <= 120:
            out.append({"text": span, "start": start, "end": start + len(span)})
    return out


def candidate_spans(text: str, hints: dict = None) -> list:
    """All answer-shaped spans in one document, with character offsets."""
    text = text or ""
    hints = hints or {}
    spans = []
    for match in QUOTED_RE.finditer(text):
        span = _clean_span(match.group(1))
        if span:
            spans.append({"text": span, "start": match.start(1),
                          "end": match.start(1) + len(match.group(1)), "kind": "quoted"})
    for match in PERCENT_RE.finditer(text):
        spans.append({"text": _clean_span(match.group(0)), "start": match.start(), "end": match.end(),
                      "kind": "quantity"})
    for match in MONEY_RE.finditer(text):
        spans.append({"text": _clean_span(match.group(0)), "start": match.start(), "end": match.end(),
                      "kind": "quantity"})
    for match in YEAR_RE.finditer(text):
        spans.append({"text": match.group(0), "start": match.start(), "end": match.end(),
                      "kind": "date"})
    for match in DATE_RE.finditer(text):
        spans.append({"text": _clean_span(match.group(0)), "start": match.start(), "end": match.end(),
                      "kind": "date"})
    for match in QUANTITY_RE.finditer(text):
        span = _clean_span(match.group(0))
        if span and re.search(r"\d", span):
            spans.append({"text": span, "start": match.start(), "end": match.end(),
                          "kind": "quantity"})
    runs = _title_candidates(text)
    for item in runs:
        spans.append({**item, "kind": "title"})
    for item in runs:
        words = [w for w in re.split(r"\s+", item["text"]) if w]
        if 2 <= len(words) <= 4 and all(_is_capitalised(w) or w.lower() in NAME_PARTICLES
                                        for w in words) and sum(len(w) > 1 for w in words) >= 2:
            spans.append({**item, "kind": "person"})
    for key in ("suffix", "prefix"):
        if hints.get(key):
            for item in _anchored_candidates(text, hints[key]):
                spans.append({**item, "kind": "title"})
    return spans


def _sentence_around(text: str, start: int, end: int) -> tuple:
    """Return (sentence_start, sentence_end) that contains the span."""
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start),
               text.rfind("\n", 0, start))
    right_candidates = [pos for pos in (text.find(".", end), text.find("!", end),
                                        text.find("?", end), text.find("\n", end)) if pos != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return max(0, left + 1), min(len(text), right + 1)


def _span_keywords(question: str) -> dict:
    lowered = (question or "").lower()
    hints = {"suffix": None, "prefix": None}
    suffix = re.search(r"ending with (?:the )?(?:word|words|letter|letters|phrase)?\s*[\"'“”]([^\"“”]{1,40})[\"'“”]",
                       lowered)
    prefix = re.search(r"(?:starting|beginning|starts|begins) with (?:the )?(?:word|words|letter|letters)?\s*"
                       r"[\"'“”]([^\"“”]{1,40})[\"'“”]", lowered)
    if suffix:
        hints["suffix"] = suffix.group(1).strip(" .,")
    if prefix:
        hints["prefix"] = prefix.group(1).strip(" .,")
    return hints


SCORE_WEIGHTS = {
    "keyword": 2,        # per question keyword in the containing sentence (capped)
    "keyword_cap": 3,
    "type": 4,           # span kind equals the expected kind
    "related": 2,        # span kind is plausible for the expected kind
    "quoted": 3,         # quoted span, titles are often quoted in text
    "rank1": 4,          # first retrieved document
    "rank3": 2,          # top three documents
    "percent": 2,        # question asks for a share and the span is a percentage
    "min_score": 6,
}


def select_answer(question: str, ranked_docs: list, text_getter, max_docs: int = 10,
                  weights: dict = None) -> dict:
    """Choose a short verbatim span; returns ``{}`` when nothing is confident."""
    weights = dict(SCORE_WEIGHTS, **(weights or {}))
    wanted = expected_type(question)
    hints = _span_keywords(question)
    terms = _keyword_terms(question)
    asks_share = bool(re.search(r"percent|percentage|share|proportion", question or "", re.I))
    if not terms:
        return {}
    best = None
    for doc_rank, doc in enumerate(list(ranked_docs)[:max_docs], 1):
        doc_id = doc["doc_id"]
        try:
            text = text_getter(doc_id) or ""
        except Exception:
            continue
        for span in candidate_spans(text, hints):
            answer = span["text"]
            floor = 1 if span["kind"] in ("quantity", "date") else 3
            if not floor <= len(answer) <= 90:
                continue
            if hints["suffix"] and not answer.lower().rstrip(".!?").endswith(hints["suffix"]):
                continue
            if hints["prefix"] and not answer.lower().startswith(hints["prefix"]):
                continue
            if span["kind"] == "quantity" and wanted != "quantity":
                continue
            if wanted == "quantity" and span["kind"] not in ("quantity", "date"):
                continue
            start, end = _sentence_around(text, span["start"], span["end"])
            sentence = text[start:end].lower()
            matched = sum(1 for term in terms if term in sentence)
            if matched == 0:
                continue
            score = weights["keyword"] * min(matched, weights["keyword_cap"])
            if span["kind"] == wanted:
                score += weights["type"]
            elif span["kind"] == "quoted" and wanted == "title":
                score += weights["type"]
            elif span["kind"] in ("title", "person") and wanted in ("title", "person", "org"):
                score += weights["related"]
            if span["kind"] == "quoted":
                score += weights["quoted"]
            if doc_rank == 1:
                score += weights["rank1"]
            elif doc_rank <= 3:
                score += weights["rank3"]
            if asks_share and "%" in answer:
                score += weights["percent"]
            key = (score, -doc_rank, -len(answer))
            if best is None or key > best[0]:
                best = (key, {"answer": answer.strip(), "doc_id": doc_id,
                              "start": span["start"], "end": span["end"],
                              "kind": span["kind"], "quote": (start, end),
                              "score": key[0], "keyword_hits": matched})
    if best is None or best[0][0] < weights["min_score"]:
        return {}
    return best[1]


def extractive_answer(question: str, quotes: list, max_chars: int = 200) -> tuple:
    """Sentence-level fallback: used when no typed span is confident enough."""
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


def make_answer(question: str, quotes: list, settings: dict, selection: dict = None) -> dict:
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
    max_chars = int(config.get("max_answer_chars") or 200)
    if selection and selection.get("answer"):
        answer = selection["answer"][:max_chars].strip()
        return {"answer": answer, "error": None,
                "note": (f"typed span ({selection.get('kind')}, score {selection.get('score')}, "
                         f"keyword hits {selection.get('keyword_hits')}) from document "
                         f"{selection.get('doc_id')}"),
                "usage": {"model_input_tokens": 0, "model_output_tokens": 0,
                          "token_notes": "No model used (extractive answer strategy)."}}
    answer, note = extractive_answer(question, quotes, max_chars)
    return {"answer": answer, "note": note, "error": None,
            "usage": {"model_input_tokens": 0, "model_output_tokens": 0,
                      "token_notes": "No model used (extractive answer strategy)."}}
