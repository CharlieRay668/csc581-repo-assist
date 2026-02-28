import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def bootstrap_ci(values: list[float], samples: int = 1000, alpha: float = 0.05, seed: int = 7) -> dict:
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None}

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(draw))
    means.sort()

    low_idx = int((alpha / 2) * (samples - 1))
    high_idx = int((1 - alpha / 2) * (samples - 1))

    return {
        "mean": mean(values),
        "ci_low": means[low_idx],
        "ci_high": means[high_idx],
    }


def pass_from_rating(row: dict, min_total: int = 6) -> bool:
    total = int(row.get("correctness", 0)) + int(row.get("grounding", 0)) + int(row.get("relevance", 0)) + int(row.get("clarity", 0))
    critical_error = bool(row.get("critical_error", False))
    return total >= min_total and not critical_error


def compute_quality_metrics(ratings: list[dict], spec: dict) -> dict:
    if not ratings:
        return {}

    min_total = int(spec["rubric"]["pass_rule"]["min_total_score"])

    task_group = defaultdict(list)
    for r in ratings:
        task_group[r["task_id"]].append(r)

    task_means = []
    task_passes = []
    critical_errors = []

    for task_rows in task_group.values():
        totals = []
        passes = []
        crits = []
        for r in task_rows:
            total = int(r.get("correctness", 0)) + int(r.get("grounding", 0)) + int(r.get("relevance", 0)) + int(r.get("clarity", 0))
            totals.append(total)
            passes.append(pass_from_rating(r, min_total=min_total))
            crits.append(bool(r.get("critical_error", False)))
        task_means.append(mean(totals))
        task_passes.append(mean([1.0 if p else 0.0 for p in passes]))
        critical_errors.append(1.0 if any(crits) else 0.0)

    by_criterion = {}
    for key in ["correctness", "grounding", "relevance", "clarity"]:
        vals = [float(r.get(key, 0)) for r in ratings]
        by_criterion[key] = {
            "mean": mean(vals) if vals else None,
            "count": len(vals),
        }

    return {
        "rubric_mean_total": mean(task_means) if task_means else None,
        "pass_rate": mean(task_passes) if task_passes else None,
        "critical_error_rate": mean(critical_errors) if critical_errors else None,
        "by_criterion": by_criterion,
        "rubric_mean_total_ci": bootstrap_ci(task_means, samples=int(spec["reporting"].get("bootstrap_samples", 1000))),
        "pass_rate_ci": bootstrap_ci(task_passes, samples=int(spec["reporting"].get("bootstrap_samples", 1000))),
    }


def compute_grounding_metrics(results: list[dict]) -> dict:
    cited_present = 0
    cited_total = 0
    overlap_vals = []
    key_claims_cited = 0
    key_claims_total = 0

    for row in results:
        g = row.get("grounding", {})
        cited_present += int(g.get("cited_spans_present", 0))
        cited_total += int(g.get("cited_spans_total", 0))
        if g.get("substring_overlap_mean") is not None:
            overlap_vals.append(float(g["substring_overlap_mean"]))
        key_claims_cited += int(g.get("key_claims_cited", 0))
        key_claims_total += int(g.get("key_claims_total", 0))

    return {
        "citation_precision": safe_div(cited_present, cited_total),
        "substring_overlap_mean": mean(overlap_vals) if overlap_vals else None,
        "source_coverage": safe_div(key_claims_cited, key_claims_total),
    }


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked: list[str], grades: dict[str, float], k: int) -> float:
    score = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        rel = float(grades.get(item, 0.0))
        score += (2 ** rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(ranked: list[str], grades: dict[str, float], k: int) -> float:
    ideal = sorted(grades.items(), key=lambda x: x[1], reverse=True)
    ideal_list = [item for item, _ in ideal]
    ideal_dcg = dcg_at_k(ideal_list, grades, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked, grades, k) / ideal_dcg


def compute_retrieval_metrics(retrieval: list[dict], k_values: list[int]) -> dict:
    if not retrieval:
        return {}

    out = {}
    for k in k_values:
        p_vals = []
        r_vals = []
        n_vals = []
        for row in retrieval:
            ranked = row.get("ranked_ids", [])
            relevant = set(row.get("relevant_ids", []))
            grades = row.get("relevance_grades", {})
            if not grades:
                grades = {doc_id: 1.0 for doc_id in relevant}

            p_vals.append(precision_at_k(ranked, relevant, k))
            r_vals.append(recall_at_k(ranked, relevant, k))
            n_vals.append(ndcg_at_k(ranked, grades, k))

        out[f"p@{k}"] = mean(p_vals) if p_vals else None
        out[f"r@{k}"] = mean(r_vals) if r_vals else None
        out[f"ndcg@{k}"] = mean(n_vals) if n_vals else None
    return out


def compute_task_success(results: list[dict], pass_k_values: list[int]) -> dict:
    if not results:
        return {}

    success = [1.0 if bool(r.get("success", False)) else 0.0 for r in results]
    exact = [1.0 if bool(r.get("exact_match", False)) else 0.0 for r in results if r.get("exact_match") is not None]
    unit = [float(r.get("unit_test_pass_rate")) for r in results if r.get("unit_test_pass_rate") is not None]

    pass_at_k = {}
    for k in pass_k_values:
        vals = []
        for r in results:
            attempts = r.get("attempts_pass")
            if attempts is None:
                continue
            vals.append(1.0 if any(bool(x) for x in attempts[:k]) else 0.0)
        pass_at_k[f"pass@{k}"] = mean(vals) if vals else None

    return {
        "success_rate": mean(success) if success else None,
        "exact_match_rate": mean(exact) if exact else None,
        "unit_test_pass_rate": mean(unit) if unit else None,
        **pass_at_k,
    }


def compute_performance(results: list[dict]) -> dict:
    if not results:
        return {}

    lat = [float(r.get("latency_ms")) for r in results if r.get("latency_ms") is not None]
    throughput = [float(r.get("throughput_tasks_per_min")) for r in results if r.get("throughput_tasks_per_min") is not None]
    cost = [float(r.get("cost_usd")) for r in results if r.get("cost_usd") is not None]
    completion = [float(r.get("completion_time_s")) for r in results if r.get("completion_time_s") is not None]

    return {
        "latency_ms": {
            "mean": mean(lat) if lat else None,
            "p50": median(lat) if lat else None,
            "p90": percentile(lat, 0.90),
            "p99": percentile(lat, 0.99),
        },
        "throughput_tasks_per_min": {
            "mean": mean(throughput) if throughput else None,
        },
        "cost_usd": {
            "mean": mean(cost) if cost else None,
            "total": sum(cost) if cost else None,
        },
        "completion_time_s": {
            "mean": mean(completion) if completion else None,
            "p90": percentile(completion, 0.90),
        },
    }


def cohen_kappa_binary(labels_a: list[int], labels_b: list[int]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None

    n = len(labels_a)
    p0 = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    pa1 = sum(labels_a) / n
    pa0 = 1.0 - pa1
    pb1 = sum(labels_b) / n
    pb0 = 1.0 - pb1

    pe = pa1 * pb1 + pa0 * pb0
    if pe == 1.0:
        return 1.0
    return (p0 - pe) / (1.0 - pe)


def weighted_kappa(scores_a: list[int], scores_b: list[int], min_score: int = 0, max_score: int = 8) -> float | None:
    if len(scores_a) != len(scores_b) or not scores_a:
        return None

    categories = list(range(min_score, max_score + 1))
    idx = {c: i for i, c in enumerate(categories)}
    n_cat = len(categories)

    obs = [[0 for _ in categories] for _ in categories]
    for a, b in zip(scores_a, scores_b):
        obs[idx[a]][idx[b]] += 1

    n = len(scores_a)
    obs = [[v / n for v in row] for row in obs]

    row_marg = [sum(row) for row in obs]
    col_marg = [sum(obs[r][c] for r in range(n_cat)) for c in range(n_cat)]

    exp = [[row_marg[i] * col_marg[j] for j in range(n_cat)] for i in range(n_cat)]

    w = [[((i - j) ** 2) / ((n_cat - 1) ** 2) for j in range(n_cat)] for i in range(n_cat)]

    obs_w = sum(w[i][j] * obs[i][j] for i in range(n_cat) for j in range(n_cat))
    exp_w = sum(w[i][j] * exp[i][j] for i in range(n_cat) for j in range(n_cat))

    if exp_w == 0:
        return 1.0
    return 1.0 - (obs_w / exp_w)


def compute_agreement(ratings: list[dict], min_total: int) -> dict:
    if not ratings:
        return {}

    grouped = defaultdict(dict)
    for r in ratings:
        task = r.get("task_id")
        rater = r.get("rater_id")
        if task is None or rater is None:
            continue
        grouped[task][rater] = r

    task_pairs = []
    for task, by_rater in grouped.items():
        raters = sorted(by_rater.keys())
        if len(raters) != 2:
            continue
        a = by_rater[raters[0]]
        b = by_rater[raters[1]]
        task_pairs.append((task, a, b))

    if not task_pairs:
        return {"note": "No exactly-two-rater overlaps found."}

    pass_a = []
    pass_b = []
    total_a = []
    total_b = []

    for _, a, b in task_pairs:
        ta = int(a.get("correctness", 0)) + int(a.get("grounding", 0)) + int(a.get("relevance", 0)) + int(a.get("clarity", 0))
        tb = int(b.get("correctness", 0)) + int(b.get("grounding", 0)) + int(b.get("relevance", 0)) + int(b.get("clarity", 0))
        total_a.append(ta)
        total_b.append(tb)
        pass_a.append(1 if pass_from_rating(a, min_total=min_total) else 0)
        pass_b.append(1 if pass_from_rating(b, min_total=min_total) else 0)

    return {
        "n_overlap_tasks": len(task_pairs),
        "cohen_kappa_pass_fail": cohen_kappa_binary(pass_a, pass_b),
        "weighted_kappa_total_score": weighted_kappa(total_a, total_b, min_score=0, max_score=8),
    }


def compare_to_baseline(current: dict, baseline: dict) -> dict:
    def flat(prefix: str, obj: dict, acc: dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat(key, v, acc)
            elif isinstance(v, (int, float)) and v is not None:
                acc[key] = float(v)
        return acc

    cur_flat = flat("", current, {})
    base_flat = flat("", baseline, {})

    deltas = {}
    for k, v in cur_flat.items():
        if k in base_flat:
            deltas[k] = v - base_flat[k]
    return deltas


def build_scorecard(spec: dict, ratings: list[dict], results: list[dict], retrieval: list[dict]) -> dict:
    min_total = int(spec["rubric"]["pass_rule"]["min_total_score"])
    k_values = [int(k) for k in spec["metrics"]["retrieval"]["k_values"]]
    pass_k_values = [int(k) for k in spec["metrics"]["task_success"]["pass_at_k"]]

    quality = compute_quality_metrics(ratings, spec)
    grounding = compute_grounding_metrics(results)
    retrieval_metrics = compute_retrieval_metrics(retrieval, k_values)
    success = compute_task_success(results, pass_k_values)
    performance = compute_performance(results)
    agreement = compute_agreement(ratings, min_total=min_total)

    return {
        "meta": {
            "spec_name": spec.get("name"),
            "n_ratings": len(ratings),
            "n_results": len(results),
            "n_retrieval": len(retrieval),
        },
        "quality": quality,
        "grounding": grounding,
        "retrieval": retrieval_metrics,
        "task_success": success,
        "performance": performance,
        "agreement": agreement,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute evaluation scorecard from JSONL inputs.")
    parser.add_argument("--spec", required=True, help="Path to eval_spec.json")
    parser.add_argument("--ratings", default=None, help="Path to ratings.jsonl")
    parser.add_argument("--results", default=None, help="Path to run_results.jsonl")
    parser.add_argument("--retrieval", default=None, help="Path to retrieval.jsonl")
    parser.add_argument("--baseline", default=None, help="Optional baseline report.json for delta comparison")
    parser.add_argument("--output", default=None, help="Optional output path for report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spec = load_json(args.spec)
    ratings = load_jsonl(args.ratings) if args.ratings else []
    results = load_jsonl(args.results) if args.results else []
    retrieval = load_jsonl(args.retrieval) if args.retrieval else []

    report = build_scorecard(spec, ratings, results, retrieval)

    if args.baseline:
        baseline = load_json(args.baseline)
        report["delta_vs_baseline"] = compare_to_baseline(report, baseline)

    out = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
