# Repo Assist — Model Evaluation Results

## Three-Way Comparison: Gemini 2.5 Flash vs. SFT Qwen3-8B vs. Base Qwen3-8B

All models evaluated on the same 100 held-out tasks against the PRFC Connect codebase, with tool access (ReAct loop, up to 8 tool turns).

---

## Summary Table

| Metric | Gemini 2.5 Flash | Qwen3-8B + SFT LoRA | Qwen3-8B (Base) |
|--------|:----------------:|:--------------------:|:---------------:|
| **Pass rate** | **0.57** | 0.48 | 0.24 |
| **Rubric mean (/8)** | **5.84** | 5.25 | 4.69 |
| **Critical error rate** | **0.03** | 0.10 | 0.23 |
| **Success rate** | **0.45** | 0.42 | 0.19 |
| **Exact match rate** | — | — | 0.17 |
| **P@3** | 0.348 | **0.413** | 0.227 |
| **R@3** | **0.528** | 0.388 | 0.203 |
| **nDCG@3** | **0.494** | 0.401 | 0.199 |

---

## Quality Rubric Breakdown (0–2 per criterion)

| Criterion | Gemini 2.5 Flash | Qwen3-8B + SFT | Qwen3-8B (Base) |
|-----------|:----------------:|:---------------:|:---------------:|
| Correctness | **1.07** | 0.87 | 0.46 |
| Grounding | **1.45** | 1.32 | 1.04 |
| Relevance | **1.75** | 1.63 | 1.66 |
| Clarity | **1.57** | 1.43 | 1.53 |

---

## Delta vs. Gemini Baseline

| Metric | SFT Δ | Base Δ |
|--------|:-----:|:------:|
| Pass rate | −0.09 | −0.33 |
| Rubric mean | −0.59 | −1.15 |
| Critical error rate | +0.07 | +0.20 |
| Success rate | −0.03 | −0.26 |
| P@3 | **+0.065** | −0.122 |
| R@3 | −0.140 | −0.325 |
| nDCG@3 | −0.093 | −0.295 |

---

## SFT Lift over Base Qwen (SFT − Base)

| Metric | Δ (SFT − Base) | Relative Improvement |
|--------|:--------------:|:--------------------:|
| Pass rate | **+0.24** | +100% |
| Rubric mean | **+0.56** | +12% |
| Critical error rate | **−0.13** | −57% |
| Success rate | **+0.23** | +121% |
| P@3 | **+0.187** | +82% |
| R@3 | **+0.185** | +91% |
| nDCG@3 | **+0.202** | +101% |
| Correctness | **+0.41** | +89% |
| Grounding | **+0.28** | +27% |

---

## Key Takeaways

1. **SFT doubles the base model's pass rate** (0.24 → 0.48), confirming that fine-tuning on Gemini trajectories teaches effective tool-calling and answer-synthesis behavior.
2. **SFT cuts critical errors by more than half** (0.23 → 0.10). The base model hallucinates at a much higher rate without the fine-tuned reasoning patterns.
3. **SFT nearly closes the gap with Gemini on success rate** (0.42 vs 0.45, only −0.03), while the base model lags far behind (0.19).
4. **SFT beats Gemini on Precision@3** (+0.065). The fine-tuned model cites fewer files but the ones it cites are more accurate.
5. **The base Qwen3-8B can use tools** (it made 1–6 tool calls per task), but without fine-tuning it doesn't use them as effectively — correctness is less than half of Gemini's (0.46 vs 1.07).
