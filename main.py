# ==============================================================================
# PROJECT: SI (Synthetic Intelligence Architecture)
# SUB-SYSTEM: Cross-Tokenizer Alignment via Vectorized Mapping & Subtoken Sync
# VERSION: v9.5.4
# CHANGELOG:
#   - v9.5.4: lp_e / lp_a are cast to .float() immediately after computation.
#             The expert and amateur model compute dtypes (fp16/bf16/4-bit) may differ,
#             which caused a dtype mismatch between mapped_am (float32) and index_put:
#             "Index put requires the source and destination dtypes
#             match, got Float for the destination and BFloat16 for the source."
#   - v9.5.3: beta clamp (0,1], proxy_valid moved to the build stage,
#             explicit dtype added to all empty(0) tensors
#   - v9.5.2: V_e == V_a check for the identity shortcut
#   - v9.5.1: proxy mask (arange < lengths), empty tensors instead of None,
#             EOS fix (fused.max()+1.0), mean-centering removed
#
# NOTE: This file is intended to run locally / on a CUDA machine with
#      `python main.py --prompt "..."`. For Colab, use the
#      `si_engine_colab.ipynb` file in the same folder (same logic, cell by cell).
# ==============================================================================
import os
import sys
import time
import math
import argparse
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SI Engine: cross-tokenizer contrastive decoding (Llama expert + Qwen amateur)."
    )
    p.add_argument("--expert-model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--amateur-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--alpha", type=float, default=0.40)
    p.add_argument("--rep-penalty", type=float, default=0.30)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--soft-limit", type=int, default=120)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.10,
                   help="Adaptive plausibility threshold. Clamped to (0, 1].")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


# ---------------- Model Loading ----------------
def load_models(expert_name, amateur_name, device, use_4bit=True):
    qc = None
    if use_4bit and device == "cuda":
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif use_4bit and device != "cuda":
        print("WARNING: 4-bit quantization requires CUDA; loading full precision instead.")

    print("\nLoading tokenizers and model weights...")
    tok_e = AutoTokenizer.from_pretrained(expert_name)
    tok_a = AutoTokenizer.from_pretrained(amateur_name)

    kw = {}
    if qc is not None:
        kw["quantization_config"] = qc
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model_e = AutoModelForCausalLM.from_pretrained(expert_name, **kw)
    model_a = AutoModelForCausalLM.from_pretrained(amateur_name, **kw)

    if qc is None:
        model_e = model_e.to(device)
        model_a = model_a.to(device)

    model_e.eval()
    model_a.eval()
    return tok_e, tok_a, model_e, model_a


# ---------------- Vocabulary Mapping ----------------
def build_vocab_map(tok_e, tok_a, model_e, model_a, device):
    """
    Returns:
        mapping        : (V_e,) long, -1 = eslesme yok
        subtoken_map   : dict eid -> List[amateur_subtoken_ids]
        proxy_ids_t    : (P,) long tensor
        proxy_matrix   : (P, L) long tensor  (pad = 0, ama maske pozisyonel)
        proxy_lengths  : (P,) long tensor
        proxy_valid    : (P, L) bool tensor  (pad pozisyonlari False)
    """
    print("\nBuilding vocabulary mapping (v4)...")

    V_e = model_e.get_output_embeddings().weight.shape[0]
    V_a = model_a.get_output_embeddings().weight.shape[0]

    # --- Identity shortcut: tokenizer vocabularies and embedding sizes must match ---
    if tok_e.get_vocab() == tok_a.get_vocab() and V_e == V_a:
        print("Same tokenizer + same embedding size -> identity mapping.")
        ids = torch.arange(V_e, dtype=torch.long, device=device)
        empty_ids = torch.empty(0, dtype=torch.long, device=device)
        empty_mat = torch.empty(0, 0, dtype=torch.long, device=device)
        empty_len = torch.empty(0, dtype=torch.long, device=device)
        empty_val = torch.empty(0, 0, dtype=torch.bool, device=device)
        return ids, {}, empty_ids, empty_mat, empty_len, empty_val

    if tok_e.get_vocab() == tok_a.get_vocab() and V_e != V_a:
        print(f"WARNING: same tokenizer but different embedding sizes "
              f"(expert={V_e}, amateur={V_a}). Falling back to explicit mapping.")

    mapping = torch.full((V_e,), -1, dtype=torch.long, device=device)
    subtoken_map = {}

    for eid in range(V_e):
        tok_str = tok_e.convert_ids_to_tokens(eid)
        if tok_str is None:
            continue
        if tok_str.startswith("<") and tok_str.endswith(">"):
            continue
        try:
            word = tok_e.convert_tokens_to_string([tok_str])
        except Exception:
            continue
        if not word:
            continue
        am_ids = tok_a.encode(word, add_special_tokens=False)
        if len(am_ids) == 1 and am_ids[0] < V_a:
            mapping[eid] = am_ids[0]
        elif len(am_ids) > 1 and all(a < V_a for a in am_ids):
            subtoken_map[eid] = am_ids

    mapped = (mapping >= 0).sum().item()
    proxy = len(subtoken_map)
    print(f"1-to-1: {mapped}, proxy: {proxy}, "
          f"covered: {mapped + proxy}/{V_e} "
          f"({100 * (mapped + proxy) / V_e:.1f}%)")

    proxy_ids = list(subtoken_map.keys())
    if proxy_ids:
        max_len = max(len(subtoken_map[e]) for e in proxy_ids)
        proxy_matrix = torch.zeros((len(proxy_ids), max_len), dtype=torch.long)
        proxy_lengths = torch.zeros(len(proxy_ids), dtype=torch.long)
        for i, e in enumerate(proxy_ids):
            subs = subtoken_map[e]
            proxy_matrix[i, :len(subs)] = torch.tensor(subs, dtype=torch.long)
            proxy_lengths[i] = len(subs)

        proxy_ids_t = torch.tensor(proxy_ids, dtype=torch.long, device=device)
        proxy_matrix = proxy_matrix.to(device)
        proxy_lengths = proxy_lengths.to(device)

        # Positional mask: computed once during the build stage
        arange = torch.arange(max_len, device=device).unsqueeze(0)  # (1, L)
        proxy_valid = arange < proxy_lengths.unsqueeze(1)           # (P, L) bool
    else:
        proxy_ids_t = torch.empty(0, dtype=torch.long, device=device)
        proxy_matrix = torch.empty(0, 0, dtype=torch.long, device=device)
        proxy_lengths = torch.empty(0, dtype=torch.long, device=device)
        proxy_valid = torch.empty(0, 0, dtype=torch.bool, device=device)

    return mapping, subtoken_map, proxy_ids_t, proxy_matrix, proxy_lengths, proxy_valid


def warm_up(model_e, model_a, tok_e, tok_a, device):
    print(f"\nWarming up {device.upper()} kernels...")
    d_e = torch.tensor([[tok_e.eos_token_id]], device=device)
    d_a = torch.tensor([[tok_a.eos_token_id]], device=device)
    with torch.no_grad():
        _ = model_e(d_e, attention_mask=torch.ones_like(d_e), use_cache=False)
        _ = model_a(d_a, attention_mask=torch.ones_like(d_a), use_cache=False)


# ---------------- Generation ----------------
@torch.no_grad()
def generate(user_query, tok_e, tok_a, model_e, model_a, expert_to_amateur_map,
             proxy_ids_t, proxy_matrix, proxy_lengths, proxy_valid, device,
             alpha=0.40, repetition_penalty=0.30, penalty_window=20,
             beta=0.10, temperature=0.0, top_p=1.0,
             soft_limit=120, absolute_max=200, verbose=True):

    # beta clamp - silently correct invalid user values
    if beta > 0:
        beta = min(max(beta, 1e-8), 1.0)

    msg_e = tok_e.apply_chat_template(
        [{"role": "user", "content": user_query}],
        tokenize=False, add_generation_prompt=True)
    msg_a = tok_a.apply_chat_template(
        [{"role": "user", "content": user_query}],
        tokenize=False, add_generation_prompt=True)

    input_e = tok_e([msg_e], return_tensors="pt").input_ids.to(device)
    input_a = tok_a([msg_a], return_tensors="pt").input_ids.to(device)

    past_e = past_a = None
    mask_e = torch.ones_like(input_e)
    mask_a = torch.ones_like(input_a)
    curr_e, curr_a = input_e, input_a

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    total = 0
    gen_ids = []
    out_parts = []

    V = expert_to_amateur_map.shape[0]
    valid_mask = expert_to_amateur_map >= 0
    mapped_am = torch.zeros(1, V, device=device)

    for step in range(absolute_max):
        out_e = model_e(input_ids=curr_e, past_key_values=past_e,
                        attention_mask=mask_e, use_cache=True)
        logits_e = out_e.logits[:, -1, :]
        past_e = out_e.past_key_values

        out_a = model_a(input_ids=curr_a, past_key_values=past_a,
                        attention_mask=mask_a, use_cache=True)
        logits_a = out_a.logits[:, -1, :]
        past_a = out_a.past_key_values

        # NOTE: the expert and amateur model compute dtypes (fp16/bf16/4-bit quantization)
        # may differ. We cast immediately after log-softmax so fusion/threshold/index_put
        # operations are always performed in float32.
        lp_e = F.log_softmax(logits_e, dim=-1).float()
        lp_a = F.log_softmax(logits_a, dim=-1).float()

        # --- Amateur log-probability mapping ---
        mapped_am.zero_()
        mapped_am[0, valid_mask] = lp_a[0, expert_to_amateur_map[valid_mask]]

        # Vectorized proxy - positional mask created during the build stage
        if proxy_ids_t.numel() > 0:
            sub_lps = lp_a[0, proxy_matrix].masked_fill(~proxy_valid, 0.0)
            proxy_means = sub_lps.sum(-1) / proxy_lengths.float()
            mapped_am[0, proxy_ids_t] = proxy_means

        # Unmapped tokens: neutral (mapped_am = 0 -> fused = lp_e)

        # --- Fusion ---
        fused = lp_e - alpha * mapped_am

        # --- Adaptive plausibility constraint ---
        if beta > 0:
            threshold = lp_e.max(dim=-1, keepdim=True).values + math.log(beta)
            fused = fused.masked_fill(lp_e < threshold, float("-inf"))

        # --- Repetition penalty (frequency-sensitive) ---
        if gen_ids:
            counts = Counter(gen_ids[-penalty_window:])
            for tid, c in counts.items():
                fused[0, tid] -= repetition_penalty * c

        # --- Force EOS to the absolute top near the end (-inf safeguard) ---
        if step >= absolute_max - 3:
            fused[0, tok_e.eos_token_id] = fused[0].max() + 1.0

        # --- Sampling or argmax ---
        if temperature > 0:
            probs = F.softmax(fused / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = sorted_probs.cumsum(-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                pick = torch.multinomial(sorted_probs, 1)
                nid = sorted_idx[0, pick].item()
            else:
                nid = torch.multinomial(probs, 1).item()
        else:
            nid = torch.argmax(fused, dim=-1).item()

        if nid == tok_e.eos_token_id:
            break

        total += 1
        gen_ids.append(nid)
        word = tok_e.decode([nid])
        if verbose:
            print(word, end="", flush=True)
        out_parts.append(word)

        # Soft limit: sentence-ending check on the cumulative generated text
        if step >= soft_limit:
            cum = "".join(out_parts).rstrip()
            if any(cum.endswith(p) for p in [".", "!", "?"]):
                break

        # --- Advance expert model ---
        curr_e = torch.tensor([[nid]], device=device)
        mask_e = torch.cat(
            [mask_e, torch.ones((1, 1), device=device, dtype=mask_e.dtype)], dim=-1)

        # --- Amateur synchronization ---
        a_id = expert_to_amateur_map[nid].item()
        if a_id >= 0:
            curr_a = torch.tensor([[a_id]], device=device)
            mask_a = torch.cat(
                [mask_a, torch.ones((1, 1), device=device, dtype=mask_a.dtype)], dim=-1)
        else:
            subs = tok_a.encode(word, add_special_tokens=False)
            if not subs:
                # Empty decode -> do not advance the amateur
                continue
            if len(subs) > 1:
                bi = torch.tensor([subs[:-1]], device=device)
                bm = torch.cat(
                    [mask_a,
                     torch.ones((1, bi.shape[-1]), device=device, dtype=mask_a.dtype)],
                    dim=-1)
                out_a2 = model_a(input_ids=bi, past_key_values=past_a,
                                 attention_mask=bm, use_cache=True)
                past_a = out_a2.past_key_values
                mask_a = bm
            curr_a = torch.tensor([[subs[-1]]], device=device)
            mask_a = torch.cat(
                [mask_a, torch.ones((1, 1), device=device, dtype=mask_a.dtype)], dim=-1)

    dur = time.time() - start
    tps = total / dur if dur > 0 else 0
    vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    print("\n" + "-" * 60)
    print(f"alpha={alpha}, beta={beta}, rep={repetition_penalty}, T={temperature}")
    print(f"{tps:.2f} tok/s | {vram:.2f} GB | {total} tokens")
    print("-" * 60)
    return "".join(out_parts)


# ---------------- CLI ----------------
def main():
    args = build_arg_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 66)
    print("SI ENGINE - v9.5.4")
    print("=" * 66)
    print(f"Active device: {device.upper()}")

    if "meta-llama" in args.expert_model.lower() and "HF_TOKEN" not in os.environ:
        print("\nNOTE: meta-llama models are gated. Run `huggingface-cli login`.")

    tok_e, tok_a, model_e, model_a = load_models(
        args.expert_model, args.amateur_model, device, use_4bit=not args.no_4bit)

    (expert_to_amateur_map, subtoken_map,
     proxy_ids_t, proxy_mat, proxy_len, proxy_val) = build_vocab_map(
        tok_e, tok_a, model_e, model_a, device)

    warm_up(model_e, model_a, tok_e, tok_a, device)

    print("=" * 66)
    print("SI INTERACTIVE CONSOLE - 'exit'/'quit' to stop")
    print("=" * 66)

    def run_one(prompt: str):
        print("\nSI Output: ", end="", flush=True)
        generate(
            prompt, tok_e, tok_a, model_e, model_a, expert_to_amateur_map,
            proxy_ids_t, proxy_mat, proxy_len, proxy_val, device,
            alpha=args.alpha, repetition_penalty=args.rep_penalty,
            penalty_window=args.window, beta=args.beta,
            temperature=args.temperature, top_p=args.top_p,
            soft_limit=args.soft_limit, absolute_max=args.max_tokens,
        )

    if args.prompt:
        run_one(args.prompt)
        sys.exit(0)

    while True:
        try:
            user_query = input("\nUser Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSI Engine: Session terminated.")
            break
        if user_query.lower() in ("exit", "quit"):
            print("\nSI Engine: Session terminated.")
            break
        if not user_query:
            continue
        run_one(user_query)


if __name__ == "__main__":
    main()
