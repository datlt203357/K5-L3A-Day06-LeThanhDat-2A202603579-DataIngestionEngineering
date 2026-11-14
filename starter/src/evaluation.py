"""Automatic retrieval metrics and per-query traces."""
import re
import unicodedata
from typing import Any


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _gold_records(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Read one or more accepted source/page/evidence gold records."""
    if question.get("gold"):
        return question["gold"]
    return [
        {
            "source_id": question["gold_source_id"],
            "page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
        }
    ]


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Mark automatic exact-evidence hits and preserve the retrieved trace.

    This exact quote check is a proxy for retrieval scoring only; it does not
    replace the required manual grounding assessment.
    """
    gold_records = _gold_records(question)
    gold_quotes = [
        _normalize_for_match(str(record.get("evidence_quote", "")))
        for record in gold_records
        if record.get("evidence_quote")
    ]

    annotated_hits = []
    for rank, hit in enumerate(hits[:3], start=1):
        matching_gold = [
            record
            for record in gold_records
            if hit.get("document_id", hit.get("source_id"))
            == record.get("source_id", record.get("gold_source_id"))
            and int(hit.get("page", -1))
            == int(record.get("page", record.get("gold_page", -2)))
        ]
        normalized_text = _normalize_for_match(str(hit.get("text", "")))
        evidence_match = any(quote in normalized_text for quote in gold_quotes)
        annotated_hits.append(
            {
                **hit,
                "rank": rank,
                "gold_source_page_match": bool(matching_gold),
                "evidence_quote_match": bool(matching_gold) and evidence_match,
            }
        )

    hit_at_1 = bool(
        annotated_hits
        and annotated_hits[0]["gold_source_page_match"]
        and annotated_hits[0]["evidence_quote_match"]
    )
    evidence_hit_at_3 = any(
        hit["gold_source_page_match"] and hit["evidence_quote_match"]
        for hit in annotated_hits
    )

    return {
        "id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold": gold_records,
        "filter": question.get("filter"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "hits": annotated_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for trace in traces:
        groups.setdefault((trace["strategy"], trace["split"]), []).append(trace)

    summaries = []
    for (strategy, split), group in sorted(groups.items()):
        count = len(group)
        hit1 = sum(bool(trace["hit_at_1"]) for trace in group)
        hit3 = sum(bool(trace["evidence_hit_at_3"]) for trace in group)
        summaries.append(
            {
                "strategy": strategy,
                "split": split,
                "questions": count,
                "hit_at_1": hit1,
                "hit_at_1_rate": hit1 / count if count else 0.0,
                "evidence_hit_at_3": hit3,
                "evidence_hit_at_3_rate": hit3 / count if count else 0.0,
            }
        )
    return summaries
