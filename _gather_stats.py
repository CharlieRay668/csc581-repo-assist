"""Gather stats on training data for documentation."""
import json

entries = []
for line in open('training_data/conversations.jsonl'):
    line = line.strip()
    if line:
        entries.append(json.loads(line))

print(f"Total conversations: {len(entries)}")

# Mode distribution
modes = {}
for e in entries:
    m = e.get('mode', '?')
    modes[m] = modes.get(m, 0) + 1
print(f"Mode distribution: {modes}")

# Message count stats
msg_counts = [len(e['messages']) for e in entries]
print(f"Messages per conversation: min={min(msg_counts)}, max={max(msg_counts)}, avg={sum(msg_counts)/len(msg_counts):.1f}")

# Count total assistant, tool_call, tool, user turns
total_sys = total_user = total_asst = total_tool = total_tc = total_final = 0
for e in entries:
    for m in e['messages']:
        r = m['role']
        if r == 'system':
            total_sys += 1
        elif r == 'user':
            total_user += 1
        elif r == 'assistant':
            total_asst += 1
            if m.get('tool_calls'):
                total_tc += 1
            else:
                total_final += 1
        elif r == 'tool':
            total_tool += 1

print(f"Total system msgs: {total_sys}")
print(f"Total user msgs: {total_user}")
print(f"Total assistant msgs: {total_asst} (tool_call: {total_tc}, final answer: {total_final})")
print(f"Total tool response msgs: {total_tool}")
print(f"Avg tool calls per conversation: {total_tc/len(entries):.1f}")

# Example tool_call message
for e in entries:
    for m in e['messages']:
        if m.get('tool_calls'):
            print(f"\nExample tool_call message:")
            print(f"  role: {m['role']}")
            print(f"  content: {m['content'][:200]}")
            print(f"  tool_calls: {json.dumps(m['tool_calls'], indent=2)[:200]}")
            break
    break

# Example tool response
for e in entries:
    for m in e['messages']:
        if m['role'] == 'tool':
            print(f"\nExample tool response message:")
            print(f"  role: {m['role']}")
            print(f"  name: {m.get('name', 'N/A')}")
            print(f"  content[:200]: {m['content'][:200]}")
            break
    break

# Example final answer
for e in entries:
    for m in e['messages']:
        if m['role'] == 'assistant' and not m.get('tool_calls'):
            print(f"\nExample final answer:")
            print(f"  role: {m['role']}")
            print(f"  content[:300]: {m['content'][:300]}")
            break
    break

# Count how many conversations have tool_calls
with_tc = sum(1 for e in entries if any(m.get('tool_calls') for m in e['messages']))
print(f"\nConversations with tool calls: {with_tc}/{len(entries)}")

# Check SFT results
import os
sft_output = "eval/prfconnect/sft_results/output.sft.json"
if os.path.exists(sft_output):
    sft = json.load(open(sft_output))
    print("\n=== SFT EVAL RESULTS ===")
    q = sft.get("quality", {})
    print(f"  Pass rate: {q.get('pass_rate')}")
    print(f"  Rubric mean: {q.get('rubric_mean_total')}")
    print(f"  Critical errors: {q.get('critical_error_rate')}")
    bc = q.get("by_criterion", {})
    for k in ["correctness", "grounding", "relevance", "clarity"]:
        v = bc.get(k, {})
        print(f"    {k}: {v.get('mean')}")
    ts = sft.get("task_success", {})
    print(f"  Success rate: {ts.get('success_rate')}")
    r = sft.get("retrieval", {})
    print(f"  P@3: {r.get('p@3')}")
    print(f"  R@3: {r.get('r@3')}")
    print(f"  nDCG@3: {r.get('ndcg@3')}")
    if "delta_vs_baseline" in sft:
        print("  Delta vs baseline (Gemini):")
        for k, v in sft["delta_vs_baseline"].items():
            sign = "+" if v >= 0 else ""
            print(f"    {k}: {sign}{v:.4f}")

# Check Gemini baseline results
baseline_output = "eval/prfconnect/real_results/output.prfc-connect.auto.json"
if os.path.exists(baseline_output):
    base = json.load(open(baseline_output))
    print("\n=== GEMINI BASELINE RESULTS ===")
    q = base.get("quality", {})
    print(f"  Pass rate: {q.get('pass_rate')}")
    print(f"  Rubric mean: {q.get('rubric_mean_total')}")
    print(f"  Critical errors: {q.get('critical_error_rate')}")
    bc = q.get("by_criterion", {})
    for k in ["correctness", "grounding", "relevance", "clarity"]:
        v = bc.get(k, {})
        print(f"    {k}: {v.get('mean')}")
    ts = base.get("task_success", {})
    print(f"  Success rate: {ts.get('success_rate')}")
    r = base.get("retrieval", {})
    print(f"  P@3: {r.get('p@3')}")
    print(f"  R@3: {r.get('r@3')}")
    print(f"  nDCG@3: {r.get('ndcg@3')}")

# Check base model results if they exist
base_output = "eval/prfconnect/base_results/output.sft.json"
if os.path.exists(base_output):
    bm = json.load(open(base_output))
    print("\n=== BASE QWEN (NO FINE-TUNING) RESULTS ===")
    q = bm.get("quality", {})
    print(f"  Pass rate: {q.get('pass_rate')}")
    print(f"  Rubric mean: {q.get('rubric_mean_total')}")
    ts = bm.get("task_success", {})
    print(f"  Success rate: {ts.get('success_rate')}")
    r = bm.get("retrieval", {})
    print(f"  P@3: {r.get('p@3')}")
