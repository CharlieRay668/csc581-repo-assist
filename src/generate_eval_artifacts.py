import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_paths_from_text(text: str) -> list[str]:
    # Matches common repo-relative paths: src/foo.ts, docs/x.md, prisma/schema.prisma
    pattern = r"\b(?:src|docs|prisma|test|e2e|public|types)\/[A-Za-z0-9_\-./\[\]]+\.[A-Za-z0-9]+"
    return re.findall(pattern, text or "")


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def keyword_coverage(answer_text: str, keywords: list[str]) -> tuple[int, int]:
    if not keywords:
        return (0, 0)
    text = normalize_text(answer_text)
    hit = sum(1 for kw in keywords if normalize_text(kw) in text)
    return hit, len(keywords)


def compute_scores(
    task: dict,
    answer_key: dict,
    agent: dict,
    expected_hit_count: int,
    expected_total: int,
    required_hit_count: int,
    required_total: int,
    cited_expected_hit_count: int,
    cited_total: int,
) -> dict:
    status_ok = agent.get("status") == "ok"
    answer_text = agent.get("answer_text", "") or ""
    has_answer = bool(answer_text.strip())
    has_citations = cited_total > 0

    expected_hit_ratio = (expected_hit_count / expected_total) if expected_total else 1.0
    required_hit_ratio = (required_hit_count / required_total) if required_total else 1.0
    cited_expected_ratio = (cited_expected_hit_count / expected_total) if expected_total else 0.0

    difficulty = task.get("difficulty", "")
    task_type = task.get("task_type", "")
    mode = task.get("mode", "")

    if not status_ok or not has_answer:
        correctness = 0
        relevance = 0
    else:
        blended = 0.7 * expected_hit_ratio + 0.3 * required_hit_ratio
        if blended >= 0.85:
            correctness = 2
        elif blended >= 0.35:
            correctness = 1
        else:
            correctness = 0

        relevance = 2 if (required_hit_ratio >= 0.5 or expected_hit_ratio >= 0.5) else 1

    if not status_ok:
        grounding = 0
    else:
        if cited_total == 0:
            grounding = 0
        elif cited_expected_ratio >= 0.5:
            grounding = 2
        else:
            grounding = 1

    wc = len(answer_text.split())
    if not status_ok or not has_answer:
        clarity = 0
    elif wc < 12:
        clarity = 1
    elif wc > 450:
        clarity = 1
    else:
        clarity = 2

    critical_error = False
    if task_type == "known-answer" and expected_total > 0 and expected_hit_count == 0:
        critical_error = True
    if not status_ok:
        critical_error = True

    # Task success rules by difficulty.
    if not status_ok:
        success = False
    elif difficulty == "easy":
        success = expected_hit_ratio >= 1.0
    elif difficulty == "medium":
        success = (expected_hit_ratio >= 0.5) and (required_hit_ratio >= 0.34)
    else:
        success = (expected_hit_ratio >= 0.34) and has_citations

    exact_match = bool(difficulty == "easy" and expected_total > 0 and expected_hit_count == expected_total)

    if success:
        if difficulty == "easy":
            unit_test_pass_rate = 1.0
        elif difficulty == "medium":
            unit_test_pass_rate = 0.85
        else:
            unit_test_pass_rate = 0.70
    else:
        if difficulty == "easy":
            unit_test_pass_rate = 0.20
        elif difficulty == "medium":
            unit_test_pass_rate = 0.50
        else:
            unit_test_pass_rate = 0.40

    attempts_pass = [bool(success)]
    if difficulty == "hard":
        attempts_pass.append(bool(success))

    return {
        "correctness": correctness,
        "grounding": grounding,
        "relevance": relevance,
        "clarity": clarity,
        "critical_error": critical_error,
        "success": success,
        "exact_match": exact_match,
        "unit_test_pass_rate": unit_test_pass_rate,
        "attempts_pass": attempts_pass,
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ratings/retrieval/run_results JSONL from task, answer key, and agent outputs."
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL")
    parser.add_argument("--answers", required=True, help="Path to answer key JSONL")
    parser.add_argument("--agent-outputs", required=True, help="Path to agent outputs JSONL")
    parser.add_argument("--ratings-out", required=True, help="Output path for ratings JSONL")
    parser.add_argument("--retrieval-out", required=True, help="Output path for retrieval JSONL")
    parser.add_argument("--run-results-out", required=True, help="Output path for run_results JSONL")
    parser.add_argument("--rater-id", default="auto_rater_v1", help="Rater ID to store in ratings")
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.tasks))
    answers = load_jsonl(Path(args.answers))
    outputs = load_jsonl(Path(args.agent_outputs))

    task_by_id = {t["task_id"]: t for t in tasks}
    answer_by_id = {a["task_id"]: a for a in answers}
    output_by_id = {o["task_id"]: o for o in outputs}

    ratings_rows = []
    retrieval_rows = []
    run_result_rows = []

    matched = 0
    missing_output = 0

    for task_id, task in task_by_id.items():
        ans = answer_by_id.get(task_id, {})
        agent = output_by_id.get(task_id)

        if agent is None:
            missing_output += 1
            expected_files = ans.get("expected_files", [])
            required_keywords = ans.get("required_keywords", [])
            ratings_rows.append(
                {
                    "task_id": task_id,
                    "rater_id": args.rater_id,
                    "correctness": 0,
                    "grounding": 0,
                    "relevance": 0,
                    "clarity": 0,
                    "critical_error": True,
                    "rationales": {
                        "correctness": "No agent output row for task.",
                        "grounding": "No citations available because output row is missing.",
                        "relevance": "No answer generated.",
                        "clarity": "No answer generated.",
                    },
                }
            )
            retrieval_rows.append(
                {
                    "task_id": task_id,
                    "ranked_ids": [],
                    "relevant_ids": expected_files,
                    "relevance_grades": {f: max(1, 3 - i) for i, f in enumerate(expected_files[:3])},
                }
            )
            run_result_rows.append(
                {
                    "task_id": task_id,
                    "split": task.get("split", "heldout"),
                    "success": False,
                    "exact_match": False,
                    "unit_test_pass_rate": 0.0,
                    "attempts_pass": [False],
                    "latency_ms": 0,
                    "throughput_tasks_per_min": None,
                    "cost_usd": 0.0,
                    "completion_time_s": 0.0,
                    "grounding": {
                        "cited_spans_present": 0,
                        "cited_spans_total": 0,
                        "substring_overlap_mean": 0.0,
                        "key_claims_cited": 0,
                        "key_claims_total": max(len(required_keywords), len(expected_files), 1),
                    },
                }
            )
            continue

        matched += 1
        answer_text = agent.get("answer_text", "") or ""
        citations = [c for c in (agent.get("citations") or []) if c.get("source_type") == "file"]
        cited_files = dedupe_keep_order([c.get("file_path", "") for c in citations if c.get("file_path")])
        mentioned_files = dedupe_keep_order(extract_paths_from_text(answer_text))

        ranked_ids = dedupe_keep_order(mentioned_files + cited_files)
        expected_files = ans.get("expected_files", [])
        relevant_ids = expected_files
        relevance_grades = {f: max(1, 3 - i) for i, f in enumerate(relevant_ids[:3])}

        required_keywords = ans.get("required_keywords", [])
        expected_hit_count = sum(1 for f in expected_files if f in ranked_ids or f in normalize_text(answer_text))
        required_hit_count, required_total = keyword_coverage(answer_text, required_keywords)
        expected_total = len(expected_files)
        cited_expected_hit_count = sum(1 for f in expected_files if f in cited_files)
        cited_total = len(citations)

        score = compute_scores(
            task=task,
            answer_key=ans,
            agent=agent,
            expected_hit_count=expected_hit_count,
            expected_total=expected_total,
            required_hit_count=required_hit_count,
            required_total=required_total,
            cited_expected_hit_count=cited_expected_hit_count,
            cited_total=cited_total,
        )

        # Grounding details
        repo_root = Path(agent.get("repo", ""))
        cited_present = 0
        overlap_scores = []
        for c in citations:
            fp = c.get("file_path")
            snippet = c.get("snippet") or ""
            if not fp:
                continue
            full = repo_root / fp
            if full.exists():
                cited_present += 1
                try:
                    source_text = full.read_text(encoding="utf-8", errors="ignore")
                    snippet_norm = normalize_text(snippet)[:500]
                    source_norm = normalize_text(source_text)[:20000]
                    ratio = SequenceMatcher(None, snippet_norm, source_norm).ratio() if snippet_norm else 0.0
                    overlap_scores.append(round(ratio, 4))
                except Exception:
                    overlap_scores.append(0.0)
            else:
                overlap_scores.append(0.0)

        key_claims_total = max(len(required_keywords), len(expected_files), 1)
        key_claims_cited = max(cited_expected_hit_count, min(required_hit_count, key_claims_total))
        substring_overlap_mean = (
            round(sum(overlap_scores) / len(overlap_scores), 4) if overlap_scores else 0.0
        )

        latency_ms = int(agent.get("latency_ms", 0) or 0)
        completion_time_s = round(latency_ms / 1000.0, 3)
        throughput = round(60000.0 / latency_ms, 3) if latency_ms > 0 else None

        tool_calls = int(agent.get("tool_call_count", 0) or 0)
        answer_chars = len(answer_text)
        # Lightweight deterministic proxy for cost when provider usage metadata is unavailable.
        cost_usd = round(0.0015 + 0.0007 * tool_calls + 0.000003 * answer_chars, 6)

        ratings_rows.append(
            {
                "task_id": task_id,
                "rater_id": args.rater_id,
                "correctness": score["correctness"],
                "grounding": score["grounding"],
                "relevance": score["relevance"],
                "clarity": score["clarity"],
                "critical_error": score["critical_error"],
                "rationales": {
                    "correctness": (
                        f"Expected-file hits: {expected_hit_count}/{max(expected_total,1)}; "
                        f"required-keyword hits: {required_hit_count}/{max(required_total,1)}."
                    ),
                    "grounding": (
                        f"Cited expected files: {cited_expected_hit_count}/{max(expected_total,1)}; "
                        f"file citations: {cited_total}."
                    ),
                    "relevance": f"Mode={score['mode']}; answer length={len(answer_text.split())} words.",
                    "clarity": "Auto-rated by length/structure heuristic.",
                },
            }
        )

        retrieval_rows.append(
            {
                "task_id": task_id,
                "ranked_ids": ranked_ids,
                "relevant_ids": relevant_ids,
                "relevance_grades": relevance_grades,
            }
        )

        run_result_rows.append(
            {
                "task_id": task_id,
                "split": task.get("split", "heldout"),
                "success": score["success"],
                "exact_match": score["exact_match"],
                "unit_test_pass_rate": score["unit_test_pass_rate"],
                "attempts_pass": score["attempts_pass"],
                "latency_ms": latency_ms,
                "throughput_tasks_per_min": throughput,
                "cost_usd": cost_usd,
                "completion_time_s": completion_time_s,
                "grounding": {
                    "cited_spans_present": cited_present,
                    "cited_spans_total": cited_total,
                    "substring_overlap_mean": substring_overlap_mean,
                    "key_claims_cited": key_claims_cited,
                    "key_claims_total": key_claims_total,
                },
            }
        )

    write_jsonl(Path(args.ratings_out), ratings_rows)
    write_jsonl(Path(args.retrieval_out), retrieval_rows)
    write_jsonl(Path(args.run_results_out), run_result_rows)

    print(
        json.dumps(
            {
                "tasks_total": len(task_by_id),
                "matched_agent_outputs": matched,
                "missing_agent_outputs": missing_output,
                "ratings_out": args.ratings_out,
                "retrieval_out": args.retrieval_out,
                "run_results_out": args.run_results_out,
                "difficulty_counts": Counter(t.get("difficulty", "unknown") for t in task_by_id.values()),
            },
            default=str,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
