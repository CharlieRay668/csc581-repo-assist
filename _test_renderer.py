"""Test qwen3 vs qwen3_disable_thinking renderer format."""
from tinker_cookbook import renderers, model_info
import tinker

service = tinker.ServiceClient()
ckpt = open("training_data/last_checkpoint.txt").read().strip()
tc = service.create_training_client_from_state(path=ckpt)
tokenizer = tc.get_tokenizer()

msgs = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

for rname in ["qwen3", "qwen3_disable_thinking"]:
    try:
        r = renderers.get_renderer(rname, tokenizer=tokenizer)
        tokens, weights = r.build_supervised_example(msgs)
        decoded = tokenizer.decode(tokens.tolist())
        idx = decoded.find("<|im_start|>assistant")
        print(f"--- {rname} ---")
        print(repr(decoded[idx:idx+120]))
        print()
    except Exception as e:
        print(f"--- {rname} ERROR: {e} ---")
