# ==============================================================================
# PROJECT: SI (Synthetic Intelligence Architecture)
# SUB-SYSTEM: Cross-Tokenizer Alignment via Vectorized Mapping & Subtoken Sync
# VERSION: v9.2 (Local / VS Code Runnable Version)
# INFRASTRUCTURE: PyTorch, Transformers, BitsAndBytes
# PLATFORM: Any machine with a CUDA-capable GPU (local, server, cloud VM)
#
# SETUP (one time):
#   python -m venv .venv
#   source .venv/bin/activate        # Windows: .venv\Scripts\activate
#   pip install transformers torch huggingface_hub bitsandbytes accelerate
#   huggingface-cli login            # needed once for gated Llama-3.2 access
#
# RUN:
#   python si_engine.py
#   python si_engine.py --alpha 0.5 --rep-penalty 0.3 --window 20
# ==============================================================================

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SI Engine: cross-tokenizer contrastive decoding (Llama expert + Qwen amateur)."
    )
    parser.add_argument("--expert-model", default="meta-llama/Llama-3.2-3B-Instruct",
                         help="HF repo id of the expert (larger) model.")
    parser.add_argument("--amateur-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                         help="HF repo id of the amateur (smaller) model.")
    parser.add_argument("--alpha", type=float, default=0.40,
                         help="Contrastive penalty strength (log-prob space).")
    parser.add_argument("--rep-penalty", type=float, default=0.30,
                         help="Repetition penalty applied within the recent token window.")
    parser.add_argument("--window", type=int, default=20,
                         help="Number of recent tokens considered for repetition penalty.")
    parser.add_argument("--soft-limit", type=int, default=100,
                         help="Token index after which generation stops at the next sentence boundary.")
    parser.add_argument("--max-tokens", type=int, default=180,
                         help="Absolute maximum number of tokens to generate.")
    parser.add_argument("--prompt", type=str, default=None,
                         help="Run a single prompt non-interactively and exit (useful for scripting/CI).")
    parser.add_argument("--no-4bit", action="store_true",
                         help="Disable 4-bit quantization (needs more VRAM, useful for debugging on CPU/MPS).")
    return parser


def load_models(expert_name: str, amateur_name: str, device: str, use_4bit: bool):
    quant_config = None
    if use_4bit and device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif use_4bit and device != "cuda":
        print("⚠️  4-bit quantization requires CUDA; loading full precision instead "
              "(this will be slow/heavy on CPU).")

    print("\n⏳ Loading tokenizers and model weights...")
    tokenizer_expert = AutoTokenizer.from_pretrained(expert_name)
    tokenizer_amateur = AutoTokenizer.from_pretrained(amateur_name)

    model_kwargs = {}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model_expert = AutoModelForCausalLM.from_pretrained(expert_name, **model_kwargs)
    model_amateur = AutoModelForCausalLM.from_pretrained(amateur_name, **model_kwargs)

    if quant_config is None:
        model_expert = model_expert.to(device)
        model_amateur = model_amateur.to(device)

    return tokenizer_expert, tokenizer_amateur, model_expert, model_amateur


def build_vocab_map(tokenizer_expert, tokenizer_amateur, model_expert, device: str) -> torch.Tensor:
    print("\n🔄 Building 1-to-1 vocabulary mapping matrix...")
    # Use the model's actual output layer size, NOT tokenizer.vocab_size
    # (Llama-3.2's tokenizer.vocab_size under-reports special/reserved tokens
    # baked into the real logits dimension -> causes an IndexError otherwise).
    expert_vocab_size = model_expert.config.vocab_size
    mapping = torch.full((expert_vocab_size,), -1, dtype=torch.long, device=device)

    for token_str, expert_id in tokenizer_expert.get_vocab().items():
        if expert_id < expert_vocab_size:
            amateur_ids = tokenizer_amateur.encode(token_str, add_special_tokens=False)
            if len(amateur_ids) == 1:
                mapping[expert_id] = amateur_ids[0]

    print("🎯 Vocabulary mapping complete.")
    return mapping


def warm_up(model_expert, model_amateur, tokenizer_expert, tokenizer_amateur, device: str):
    print(f"\n🔥 Warming up {device.upper()} kernels (for accurate benchmark timing)...")
    dummy_expert = torch.tensor([[tokenizer_expert.eos_token_id]], device=device)
    dummy_amateur = torch.tensor([[tokenizer_amateur.eos_token_id]], device=device)
    with torch.no_grad():
        _ = model_expert(dummy_expert, use_cache=False)
        _ = model_amateur(dummy_amateur, use_cache=False)


def generate(user_query, tokenizer_expert, tokenizer_amateur, model_expert, model_amateur,
             expert_to_amateur_map, device, alpha, repetition_penalty, penalty_window,
             soft_limit, absolute_max):

    msg_expert = tokenizer_expert.apply_chat_template(
        [{"role": "user", "content": user_query}], tokenize=False, add_generation_prompt=True)
    msg_amateur = tokenizer_amateur.apply_chat_template(
        [{"role": "user", "content": user_query}], tokenize=False, add_generation_prompt=True)

    input_ids_expert = tokenizer_expert([msg_expert], return_tensors="pt").input_ids.to(device)
    input_ids_amateur = tokenizer_amateur([msg_amateur], return_tensors="pt").input_ids.to(device)

    past_expert, past_amateur = None, None
    mask_expert = torch.ones_like(input_ids_expert)
    mask_amateur = torch.ones_like(input_ids_amateur)

    curr_expert = input_ids_expert
    curr_amateur = input_ids_amateur

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    total_tokens_generated = 0
    generated_token_ids = []
    output_text_parts = []

    for token_index in range(absolute_max):
        with torch.no_grad():
            out_expert = model_expert(input_ids=curr_expert, past_key_values=past_expert,
                                       attention_mask=mask_expert, use_cache=True)
            logits_expert = out_expert.logits[:, -1, :]
            past_expert = out_expert.past_key_values

            out_amateur = model_amateur(input_ids=curr_amateur, past_key_values=past_amateur,
                                         attention_mask=mask_amateur, use_cache=True)
            logits_amateur = out_amateur.logits[:, -1, :]
            past_amateur = out_amateur.past_key_values

            log_probs_expert = F.log_softmax(logits_expert, dim=-1)
            log_probs_amateur = F.log_softmax(logits_amateur, dim=-1)

            mapped_amateur_log_probs = torch.zeros_like(log_probs_expert)
            valid_mask = expert_to_amateur_map >= 0
            mapped_amateur_log_probs[0, valid_mask] = log_probs_amateur[0, expert_to_amateur_map[valid_mask]]

            fused_log_probs = log_probs_expert - (alpha * mapped_amateur_log_probs)

            if generated_token_ids:
                recent_ids = set(generated_token_ids[-penalty_window:])
                for tid in recent_ids:
                    fused_log_probs[0, tid] -= repetition_penalty

            if token_index >= (absolute_max - 5):
                fused_log_probs[0, tokenizer_expert.eos_token_id] += 100.0

            next_token = torch.argmax(fused_log_probs, dim=-1, keepdim=True)
            total_tokens_generated += 1
            generated_token_ids.append(next_token.item())

            chosen_word = tokenizer_expert.decode([next_token.item()])
            print(chosen_word, end="", flush=True)
            output_text_parts.append(chosen_word)

            if token_index >= soft_limit and any(p in chosen_word for p in [".", "!", "?"]):
                break
            if next_token.item() == tokenizer_expert.eos_token_id:
                break

            curr_expert = next_token
            mask_expert = torch.cat([mask_expert, torch.ones((1, 1), device=device, dtype=mask_expert.dtype)], dim=-1)

            amateur_token_id = expert_to_amateur_map[next_token.item()].item()
            if amateur_token_id >= 0:
                curr_amateur = torch.tensor([[amateur_token_id]], device=device)
                mask_amateur = torch.cat([mask_amateur, torch.ones((1, 1), device=device, dtype=mask_amateur.dtype)], dim=-1)
            else:
                amateur_sub_tokens = tokenizer_amateur.encode(chosen_word, add_special_tokens=False)
                if amateur_sub_tokens:
                    if len(amateur_sub_tokens) > 1:
                        batch_input = torch.tensor([amateur_sub_tokens[:-1]], device=device)
                        batch_mask = torch.cat(
                            [mask_amateur, torch.ones((1, batch_input.shape[-1]), device=device, dtype=mask_amateur.dtype)],
                            dim=-1)
                        out_amateur = model_amateur(input_ids=batch_input, past_key_values=past_amateur,
                                                     attention_mask=batch_mask, use_cache=True)
                        past_amateur = out_amateur.past_key_values
                        mask_amateur = batch_mask

                    curr_amateur = torch.tensor([[amateur_sub_tokens[-1]]], device=device)
                    mask_amateur = torch.cat([mask_amateur, torch.ones((1, 1), device=device, dtype=mask_amateur.dtype)], dim=-1)
                else:
                    curr_amateur = torch.tensor([[tokenizer_amateur.eos_token_id]], device=device)
                    mask_amateur = torch.cat([mask_amateur, torch.ones((1, 1), device=device, dtype=mask_amateur.dtype)], dim=-1)

    end_time = time.time()
    duration = end_time - start_time
    tokens_per_sec = total_tokens_generated / duration if duration > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    print("\n" + "-" * 60)
    print("📊 BENCHMARK METRICS")
    print(f"  • Config            : alpha={alpha}, rep_penalty={repetition_penalty}, window={penalty_window}")
    print(f"  • Generation Speed  : {tokens_per_sec:.2f} tokens/second")
    print(f"  • Peak VRAM Usage   : {peak_vram:.2f} GB")
    print("-" * 60)

    return "".join(output_text_parts)


def main():
    args = build_arg_parser().parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("==================================================================")
    print("🧠 SI (SYNTHETIC INTELLIGENCE) ENGINE — LOCAL / VS CODE VERSION")
    print("==================================================================")
    print(f"🚀 Active device: {device.upper()}")

    if "meta-llama" in args.expert_model.lower() and "HF_TOKEN" not in os.environ:
        print("\n📢 NOTE: meta-llama models are gated on Hugging Face.")
        print("Run `huggingface-cli login` once, or set the HF_TOKEN environment variable,")
        print("and make sure your HF account has been granted access to this model.")

    tokenizer_expert, tokenizer_amateur, model_expert, model_amateur = load_models(
        args.expert_model, args.amateur_model, device, use_4bit=not args.no_4bit
    )

    expert_to_amateur_map = build_vocab_map(tokenizer_expert, tokenizer_amateur, model_expert, device)
    warm_up(model_expert, model_amateur, tokenizer_expert, tokenizer_amateur, device)

    print("==================================================================")
    print("💬 SI INTERACTIVE CONSOLE  (English prompts recommended — see README)")
    print("Type 'exit' or 'quit' to stop.")
    print("==================================================================")

    def run_one(prompt: str):
        print("\n🤖 SI Output: ", end="", flush=True)
        generate(
            prompt, tokenizer_expert, tokenizer_amateur, model_expert, model_amateur,
            expert_to_amateur_map, device,
            alpha=args.alpha, repetition_penalty=args.rep_penalty, penalty_window=args.window,
            soft_limit=args.soft_limit, absolute_max=args.max_tokens,
        )

    if args.prompt:
        # Non-interactive single-shot mode, e.g. `python si_engine.py --prompt "..."`
        run_one(args.prompt)
        sys.exit(0)

    while True:
        try:
            user_query = input("\n👤 User Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🤖 SI Engine: Session terminated. Goodbye!")
            break

        if user_query.lower() in ("exit", "quit"):
            print("\n🤖 SI Engine: Session terminated. Goodbye!")
            break
        if not user_query:
            continue

        run_one(user_query)


if __name__ == "__main__":
    main()
