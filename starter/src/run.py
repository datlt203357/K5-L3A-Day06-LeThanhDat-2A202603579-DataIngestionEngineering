"""Run ingestion, chunking, embedding, vector search, and automatic metrics.

Run from ``starter/`` with ``python -m src.run``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore

logger = logging.getLogger(__name__)

FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 50
PARAGRAPH_MAX_SIZE = 500
MAX_EMBEDDING_TOKENS = 120


def _expand_page_ranges(page_ranges: list[int | str]) -> set[int]:
    """Expand manifest entries such as 2, "5-8" into 1-based page numbers."""
    pages: set[int] = set()
    for entry in page_ranges:
        if isinstance(entry, int):
            start = end = entry
        elif isinstance(entry, str):
            match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", entry)
            if not match:
                raise ValueError(f"Invalid included_pdf_pages entry: {entry!r}")
            start = int(match.group(1))
            end = int(match.group(2) or start)
        else:
            raise TypeError(f"Unsupported page range entry: {entry!r}")

        if start < 1 or end < start:
            raise ValueError(f"Invalid PDF page range: {entry!r}")
        pages.update(range(start, end + 1))
    return pages


def load_corpus() -> tuple[list[DocumentPage], dict[str, Any]]:
    """Load the single PDF source described by the repository manifest."""
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if len(sources) != 1:
        raise ValueError(
            f"Expected exactly one source in {manifest_path}, found {len(sources)}"
        )
    source_info = sources[0]

    pdf_path = Path(source_info["source_path_or_url"])
    if not pdf_path.is_absolute():
        pdf_path = repo_root / pdf_path
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Corpus PDF not found: {pdf_path}")

    physical_page_count = int(source_info["pdf_physical_page_count"])
    included_pages = _expand_page_ranges(source_info["included_pdf_pages"])
    excluded_pages = {
        int(entry["pdf_page"]) for entry in source_info["excluded_pdf_pages"]
    }
    all_pages = set(range(1, physical_page_count + 1))

    if included_pages & excluded_pages:
        raise ValueError("A PDF page is listed as both included and excluded")
    if included_pages | excluded_pages != all_pages:
        missing = sorted(all_pages - included_pages - excluded_pages)
        extra = sorted((included_pages | excluded_pages) - all_pages)
        raise ValueError(
            "Manifest page lists must account for every physical PDF page; "
            f"unlisted={missing}, out_of_range={extra}"
        )
    if len(included_pages) < 30:
        raise ValueError(
            f"Corpus has only {len(included_pages)} included pages; at least 30 are required"
        )

    pages = load_pdf(
        path=pdf_path,
        document_id=source_info["source_id"],
        source=source_info["source_path_or_url"],
        sections=source_info.get("sections", []),
        excluded_pages=excluded_pages,
        text_corrections=source_info.get("extraction", {}).get(
            "manual_corrections", []
        ),
    )

    loaded_page_numbers = {page.page for page in pages}
    if loaded_page_numbers != included_pages:
        missing = sorted(included_pages - loaded_page_numbers)
        unexpected = sorted(loaded_page_numbers - included_pages)
        raise ValueError(
            f"PDF pages do not match manifest; missing={missing}, unexpected={unexpected}"
        )

    return pages, source_info


def build_strategy_chunks(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    """Build the A/B chunkings; C reuses B's content with distinct stable IDs."""
    chunks_a = [
        chunk
        for page in pages
        for chunk in fixed_size(page, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
    ]
    chunks_b = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, PARAGRAPH_MAX_SIZE)
    ]
    chunks_c = [
        replace(
            chunk,
            chunk_id=chunk.chunk_id.replace(":paragraph:", ":paragraph_c:"),
            metadata=dict(chunk.metadata),
        )
        for chunk in chunks_b
    ]
    return {"A": chunks_a, "B": chunks_b, "C": chunks_c}


def load_questions(repo_root: Path) -> list[dict[str, Any]]:
    """Load and validate the frozen dev/test benchmark file."""
    questions_path = repo_root / "benchmark" / "questions.jsonl"
    if not questions_path.is_file():
        raise FileNotFoundError(f"Benchmark questions not found: {questions_path}")

    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(questions) < 20:
        raise ValueError(f"Benchmark has {len(questions)} questions; at least 20 required")

    split_counts = Counter(question.get("split") for question in questions)
    if split_counts.get("dev", 0) < 15 or split_counts.get("test", 0) < 5:
        raise ValueError(
            "Benchmark requires at least 15 dev and 5 test questions; "
            f"found dev={split_counts.get('dev', 0)}, test={split_counts.get('test', 0)}"
        )

    filtered = [question for question in questions if question.get("filter")]
    filtered_test = [question for question in filtered if question["split"] == "test"]
    if len(filtered) < 3 or not filtered_test:
        raise ValueError("Benchmark requires at least 3 filtered questions including a test question")
    return questions


def _token_length_report(embedder: MyEmbedder, chunks: list[Chunk]) -> tuple[int, int]:
    """Return the number of over-limit chunks and maximum untruncated token length."""
    if not chunks:
        return 0, 0
    tokenizer = embedder.model.tokenizer
    encoded = tokenizer(
        [chunk.text for chunk in chunks],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
    limit = int(embedder.model.max_seq_length)
    return sum(length > limit for length in lengths), max(lengths)


def _split_to_token_budget(text: str, tokenizer: Any, token_budget: int) -> list[str]:
    """Recursively split an oversized chunk at whitespace boundaries."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    if len(encoded["input_ids"]) <= token_budget:
        return [text]

    midpoint = len(text) // 2
    cut = text.rfind(" ", 1, midpoint + 1)
    if cut <= 0:
        cut = text.find(" ", midpoint)
    if cut <= 0 or cut >= len(text) - 1:
        raise ValueError(
            "A single token exceeds the embedding token budget; inspect this chunk manually"
        )

    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        raise ValueError("Unable to split an oversized chunk at a whitespace boundary")
    return _split_to_token_budget(left, tokenizer, token_budget) + _split_to_token_budget(
        right, tokenizer, token_budget
    )


def _enforce_token_budget(
    chunks: list[Chunk],
    tokenizer: Any,
    token_budget: int,
) -> list[Chunk]:
    """Keep all embedded inputs below the model limit without silent truncation."""
    safe_chunks = []
    for chunk in chunks:
        pieces = _split_to_token_budget(chunk.text, tokenizer, token_budget)
        if len(pieces) == 1:
            safe_chunks.append(chunk)
            continue

        for index, piece in enumerate(pieces):
            metadata = dict(chunk.metadata)
            metadata["parent_chunk_id"] = chunk.chunk_id
            metadata["token_safe_split"] = True
            safe_chunks.append(
                replace(
                    chunk,
                    chunk_id=f"{chunk.chunk_id}:token:{index:03d}",
                    text=piece,
                    metadata=metadata,
                )
            )
    return safe_chunks


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the same retrieval benchmark against A, B, and C."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_time = datetime.now(timezone.utc)
    run_started_at = run_time.isoformat()
    run_id = run_time.strftime("%Y%m%dT%H%M%SZ")
    timer_start = perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / "starter" / ".env")

    pages, source_info = load_corpus()
    ingestion_seconds = perf_counter() - timer_start

    section_counts = Counter(
        page.metadata.get("section", "unassigned") for page in pages
    )
    empty_pages = [page.page for page in pages if not page.text]
    extraction_info = source_info.get("extraction", {})
    expected_counts = {
        int(page): int(count)
        for page, count in extraction_info.get("page_character_counts", {}).items()
    }
    count_mismatches = []
    for page in pages:
        if page.page not in expected_counts:
            continue
        actual_count = len(re.sub(r"\s+", " ", page.text).strip())
        if actual_count != expected_counts[page.page]:
            count_mismatches.append(
                (page.page, expected_counts[page.page], actual_count)
            )

    print(f"Source: {source_info['title']}")
    print(f"PDF pages loaded: {len(pages)}")
    print(f"Pages with no extracted text: {empty_pages or 'none'}")
    print("Pages by section:")
    for section_id, count in sorted(section_counts.items()):
        print(f"  {section_id}: {count}")
    if count_mismatches:
        print("Character-count differences (page, manifest, current extraction):")
        for mismatch in count_mismatches:
            print(f"  {mismatch[0]}: {mismatch[1]} vs {mismatch[2]}")
    else:
        print("Manifest character counts: match")

    chunk_timer = perf_counter()
    strategy_chunks = build_strategy_chunks(pages)
    chunking_seconds = perf_counter() - chunk_timer
    print("Chunking configurations:")
    print(
        f"  A: fixed-size, size={FIXED_CHUNK_SIZE}, "
        f"overlap={FIXED_CHUNK_OVERLAP} characters"
    )
    print(f"  B: paragraph-aware, max_size={PARAGRAPH_MAX_SIZE} characters")
    print("  C: same paragraph chunks as B; metadata filter is applied at search time")
    for strategy, chunks in strategy_chunks.items():
        lengths = [len(chunk.text) for chunk in chunks]
        print(
            f"  {strategy}: chunks={len(chunks)}, "
            f"mean_chars={mean(lengths):.1f}, median_chars={median(lengths):.1f}, "
            f"max_chars={max(lengths)}"
            if lengths
            else f"  {strategy}: chunks=0"
        )
        missing_sections = sum("section" not in chunk.metadata for chunk in chunks)
        if missing_sections:
            print(f"    chunks missing section metadata: {missing_sections}")

    model_name = os.getenv("EMBEDDING_MODEL")
    database_url = os.getenv("DATABASE_URL")
    if not model_name:
        raise ValueError("EMBEDDING_MODEL is missing from starter/.env")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from starter/.env")

    questions = load_questions(repo_root)
    embedder = MyEmbedder(model_name)
    if embedder.dimensions != 384:
        raise ValueError(
            f"Configured pgvector schema expects 384 dimensions; model gives {embedder.dimensions}"
        )

    # Keep every input below the model's 128-token limit. This preserves all
    # source text instead of relying on the encoder's silent truncation.
    strategy_chunks["A"] = _enforce_token_budget(
        strategy_chunks["A"], embedder.model.tokenizer, MAX_EMBEDDING_TOKENS
    )
    strategy_chunks["B"] = _enforce_token_budget(
        strategy_chunks["B"], embedder.model.tokenizer, MAX_EMBEDDING_TOKENS
    )
    strategy_chunks["C"] = [
        replace(
            chunk,
            chunk_id=chunk.chunk_id.replace(":paragraph:", ":paragraph_c:"),
            metadata=dict(chunk.metadata),
        )
        for chunk in strategy_chunks["B"]
    ]

    token_over_limit = {}
    for strategy in ("A", "B"):
        over_limit, max_tokens = _token_length_report(embedder, strategy_chunks[strategy])
        token_over_limit[strategy] = over_limit
        print(
            f"Strategy {strategy} token lengths: max={max_tokens}, "
            f"over model limit ({embedder.model.max_seq_length})={over_limit}"
        )
    if any(token_over_limit.values()):
        raise ValueError(
            "Some chunks exceed the embedding model token limit and would be truncated. "
            "Reduce chunk sizes before creating benchmark results."
        )

    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    store.create_schema()

    embedding_timer = perf_counter()
    vectors_a = embedder.embed_documents(
        [chunk.text for chunk in strategy_chunks["A"]]
    )
    vectors_b = embedder.embed_documents(
        [chunk.text for chunk in strategy_chunks["B"]]
    )
    # C has the same chunk text as B; reuse those vectors and only change metadata-filter behavior.
    store.upsert("A", strategy_chunks["A"], vectors_a)
    store.upsert("B", strategy_chunks["B"], vectors_b)
    store.upsert("C", strategy_chunks["C"], vectors_b)
    embedding_and_store_seconds = perf_counter() - embedding_timer

    retrieval_timer = perf_counter()
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    traces: list[dict[str, Any]] = []
    for strategy in ("A", "B", "C"):
        for question in questions:
            query_filter = question.get("filter") if strategy == "C" else None
            hits = store.search(
                strategy=strategy,
                query_vector=query_vectors[question["id"]],
                top_k=3,
                filters=query_filter,
            )
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            traces.append(trace)
    retrieval_seconds = perf_counter() - retrieval_timer

    summaries = summarize(traces)
    output_dir = repo_root / "results" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "retrieval_traces.jsonl", traces)
    (output_dir / "retrieval_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_metadata = {
        "run_started_at_utc": run_started_at,
        "corpus_id": source_info.get("source_id"),
        "embedding_model": model_name,
        "embedding_dimensions": embedder.dimensions,
        "metric": "cosine",
        "top_k": 3,
        "chunking": {
            "A": {"size_chars": FIXED_CHUNK_SIZE, "overlap_chars": FIXED_CHUNK_OVERLAP},
            "B": {"method": "paragraph-aware", "max_size_chars": PARAGRAPH_MAX_SIZE},
            "C": {"same_chunks_as": "B", "filter": "metadata.section when provided"},
        },
        "questions": len(questions),
        "dev_questions": sum(q["split"] == "dev" for q in questions),
        "test_questions": sum(q["split"] == "test" for q in questions),
        "token_over_limit_chunks": token_over_limit,
        "max_embedding_tokens": MAX_EMBEDDING_TOKENS,
        "chunk_counts": {
            strategy: len(chunks) for strategy, chunks in strategy_chunks.items()
        },
        "timing_seconds": {
            "ingestion": round(ingestion_seconds, 3),
            "chunking": round(chunking_seconds, 3),
            "embedding_and_store": round(embedding_and_store_seconds, 3),
            "retrieval": round(retrieval_seconds, 3),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Retrieval metrics (automatic exact-evidence proxy):")
    for row in summaries:
        print(
            f"  {row['strategy']} {row['split']}: "
            f"Hit@1={row['hit_at_1']}/{row['questions']}, "
            f"Evidence Hit@3={row['evidence_hit_at_3']}/{row['questions']}"
        )

    filter_questions = [question for question in questions if question.get("filter")]
    c_filter_traces = {
        trace["id"]: trace
        for trace in traces
        if trace["strategy"] == "C"
        and next(q for q in questions if q["id"] == trace["id"]).get("filter")
    }
    filter_compliant = 0
    for question in filter_questions:
        trace = c_filter_traces[question["id"]]
        expected_filter = question["filter"]
        returned = trace["hits"]
        if returned and all(
            all(hit.get("metadata", {}).get(key) == value for key, value in expected_filter.items())
            for hit in returned
        ):
            filter_compliant += 1
    print(
        f"C filter compliance: {filter_compliant}/{len(filter_questions)} filtered questions"
    )
    print(f"Full traces and summaries written to {output_dir}")
    print(
        "Automatic quote matching is only a retrieval proxy. Complete the manual "
        "coherence, metadata, and grounding audits before drawing conclusions."
    )


if __name__ == "__main__":
    main()
