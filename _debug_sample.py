"""Debug: test Tinker sampling to understand the output format."""
import tinker
from tinker import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

CKPT = "tinker://38b4be07-1ac0-5d16-81f6-73c8918aca6e:train:0/weights/sft-epoch-7"

print("Connecting to Tinker...")
service = tinker.ServiceClient()

# First, persist the checkpoint (remove TTL)
print("Persisting checkpoint (removing TTL)...")
rest = service.create_rest_client()
rest.set_checkpoint_ttl_from_tinker_path(CKPT, None).result()
print("  Checkpoint persisted (no expiration).")

# Save path to file for future use
from pathlib import Path
Path("training_data/last_checkpoint.txt").write_text(CKPT + "\n")
print(f"  Saved to training_data/last_checkpoint.txt")

# Create sampling client: load checkpoint into a training client, then get sampler
print("Loading checkpoint into training client...")
tc = service.create_training_client_from_state(path=CKPT)
print("Getting sampling client from loaded weights...")
sampling_client = tc.save_weights_and_get_sampling_client()
print("Sampling client ready.")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

# Test with pre-filled think block (the fix for thinking mode)
prompt_text = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n</think>\n\n"
)
tokens = tokenizer.encode(prompt_text)
prompt = types.ModelInput.from_ints(tokens=tokens)

params = types.SamplingParams(max_tokens=200, temperature=0.3, stop=["<|im_end|>"])

print(f"Prompt tokens: {len(tokens)}")
print("Sampling (pre-filled think block)...")
result = sampling_client.sample(prompt, 1, params).result()

import re
seq = result.sequences[0]
print(f"stop_reason: {seq.stop_reason}")
print(f"tokens count: {len(seq.tokens)}")
decoded = tokenizer.decode(seq.tokens).strip()
# Strip any residual think tags
decoded = re.sub(r'<think>.*?</think>', '', decoded, flags=re.DOTALL).strip()
decoded = re.sub(r'<think>.*', '', decoded, flags=re.DOTALL).strip()
for st in ["<|im_end|>", "<|endoftext|>"]:
    if decoded.endswith(st):
        decoded = decoded[:-len(st)].strip()
print(f"Decoded ({len(decoded.split())} words): {decoded}")

# Also test with a repo-style question
print("\n--- Repo-style question ---")
prompt2_text = (
    "<|im_start|>system\nYou are an expert repository assistant. "
    "Mode: LOCATE. Identify exactly which files and line ranges implement the requested functionality.\n"
    "Always cite specific files and line numbers.<|im_end|>\n"
    "<|im_start|>user\nHow does authentication work in this codebase?<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n</think>\n\n"
)
tokens2 = tokenizer.encode(prompt2_text)
prompt2 = types.ModelInput.from_ints(tokens=tokens2)
params2 = types.SamplingParams(max_tokens=500, temperature=0.3, stop=["<|im_end|>"])
print(f"Prompt tokens: {len(tokens2)}")
print("Sampling...")
result2 = sampling_client.sample(prompt2, 1, params2).result()
seq2 = result2.sequences[0]
decoded2 = tokenizer.decode(seq2.tokens).strip()
decoded2 = re.sub(r'<think>.*?</think>', '', decoded2, flags=re.DOTALL).strip()
decoded2 = re.sub(r'<think>.*', '', decoded2, flags=re.DOTALL).strip()
for st in ["<|im_end|>", "<|endoftext|>"]:
    if decoded2.endswith(st):
        decoded2 = decoded2[:-len(st)].strip()
print(f"stop_reason: {seq2.stop_reason}")
print(f"Decoded ({len(decoded2.split())} words):")
print(decoded2[:500])
