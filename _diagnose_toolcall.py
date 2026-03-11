"""Diagnostic: compare training prompt format vs eval prompt format."""
import json, os, sys
import tinker
from tinker import types as ttypes
from tinker_cookbook import renderers, model_info

from src.model_backend import _build_qwen_tool_block, TOOL_SCHEMAS, _parse_qwen_tool_calls

# Load checkpoint
ckpt = open("training_data/last_checkpoint.txt").read().strip()
print(f"Checkpoint: {ckpt}")

service = tinker.ServiceClient()
tc = service.create_training_client_from_state(path=ckpt)
tokenizer = tc.get_tokenizer()
sc = tc.save_weights_and_get_sampling_client()

tools = TOOL_SCHEMAS
tool_block = _build_qwen_tool_block(tools)

system_msg = (
    "You are an expert repository assistant. "
    "Mode: LOCATE. Identify exactly which files and line ranges implement the requested functionality. "
    "Be concise — list locations first, brief explanation second.\n"
    "Always cite specific files and line numbers."
)
question = "Where is the POST /api/referrals handler implemented?"
combined_system = tool_block + "\n\n" + system_msg

# === Test 1: With <think> prefill (current eval behavior) ===
print("\n=== TEST 1: WITH <think> PREFILL ===")
prompt_parts = [
    f"<|im_start|>system\n{combined_system}<|im_end|>",
    f"<|im_start|>user\n{question}<|im_end|>",
    "<|im_start|>assistant\n<think>\n</think>\n\n",
]
prompt_text = "\n".join(prompt_parts)
prompt_tokens = tokenizer.encode(prompt_text)
print(f"Prompt tokens: {len(prompt_tokens)}")

prompt = ttypes.ModelInput.from_ints(tokens=prompt_tokens)
params = ttypes.SamplingParams(max_tokens=512, temperature=0.3, stop=["<|im_end|>"])
result = sc.sample(prompt, 1, params).result()
raw1 = tokenizer.decode(result.sequences[0].tokens).strip()
print(f"RAW OUTPUT: {repr(raw1[:600])}")
_, tcs1 = _parse_qwen_tool_calls(raw1)
print(f"Tool calls found: {len(tcs1)}")
for t in tcs1:
    print(f"  {t.name}({t.arguments})")

# === Test 2: Without <think> prefill ===
print("\n=== TEST 2: WITHOUT <think> PREFILL ===")
prompt_parts2 = [
    f"<|im_start|>system\n{combined_system}<|im_end|>",
    f"<|im_start|>user\n{question}<|im_end|>",
    "<|im_start|>assistant\n",
]
prompt_text2 = "\n".join(prompt_parts2)
prompt_tokens2 = tokenizer.encode(prompt_text2)
prompt2 = ttypes.ModelInput.from_ints(tokens=prompt_tokens2)
result2 = sc.sample(prompt2, 1, params).result()
raw2 = tokenizer.decode(result2.sequences[0].tokens).strip()
print(f"RAW OUTPUT: {repr(raw2[:600])}")
_, tcs2 = _parse_qwen_tool_calls(raw2)
print(f"Tool calls found: {len(tcs2)}")
for t in tcs2:
    print(f"  {t.name}({t.arguments})")

# === Test 3: Check what the renderer produces ===
print("\n=== TEST 3: RENDERER FORMAT CHECK ===")
renderer_name = model_info.get_recommended_renderer_name("Qwen/Qwen3-8B")
print(f"Recommended renderer: {renderer_name}")
renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

train_msgs = [
    {"role": "system", "content": combined_system},
    {"role": "user", "content": question},
    {"role": "assistant", "content": '<tool_call>\n{"name": "search_repo", "arguments": {"query": "POST /api/referrals"}}\n</tool_call>'},
]
tokens, weights = renderer.build_supervised_example(train_msgs)
decoded = tokenizer.decode(tokens.tolist())
idx = decoded.find("<|im_start|>assistant")
if idx >= 0:
    print(f"Renderer assistant format:\n{repr(decoded[idx:idx+250])}")

# Show our manual format for comparison
manual_assistant = "<|im_start|>assistant\n<think>\n</think>\n\n"
print(f"\nManual eval prefix:\n{repr(manual_assistant)}")

# Check if renderer includes think tags
if "<think>" in decoded:
    print("\nRenderer INCLUDES <think> tags")
else:
    print("\nRenderer does NOT include <think> tags")
