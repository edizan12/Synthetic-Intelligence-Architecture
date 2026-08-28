SI: Cross-Tokenizer Contrastive Decoding

Combine two LLMs with different tokenizers at inference time to reduce hallucinations — no fine-tuning, no external judge/agent calls, runs on a free-tier Colab T4 GPU.

The idea

Most hallucination-mitigation setups today use a text-level multi-agent pattern: one model generates an answer, another model (or the same one) reads and critiques it via a second API call. That works, but it's slow, expensive in tokens, and only ever looks at the final decoded text — never at the actual probability distribution the model had in mind.

This project instead operates at the logit level, before any text is decoded:

prompt
  │
  ├──► Llama-3.2-3B (expert)  ──► logits_E
  └──► Qwen2.5-1.5B (amateur) ──► logits_A
                    │
                    ▼
     fused = log_softmax(logits_E) − α · log_softmax(logits_A)
                    │
                    ▼
              next token (argmax)

The intuition, borrowed from contrastive decoding (Li et al., 2022) and DoLa-style methods: tokens that the smaller/weaker model is also confident about are often generic, high-frequency, or "safe" completions — not necessarily wrong, but not where the expert model's distinctive knowledge shows up. Subtracting a scaled version of the amateur's distribution nudges the output toward the expert model's more specific, differentiated probability mass.

Important framing note: the two models differ in scale (3B vs 1.5B) and training data, not in some cleanly separable "Western" vs "Eastern" worldview. Early informal testing suggests this setup makes the model more willing to say "I don't know" / "I can't verify this" instead of confabulating an answer (see examples below) — but this is a hypothesis under test, not a proven bias-correction mechanism. Treat any claims about "neutralizing cultural bias" with skepticism; that framing isn't rigorously established here.

Why cross-tokenizer is the hard part

Llama and Qwen don't share a vocabulary. Naively comparing their logit tensors index-by-index is meaningless — index 4021 in one tokenizer and the other refer to unrelated tokens. This project builds a string-based one-to-one vocabulary mapping: for every expert (Llama) token, we check whether it decodes to a string that re-encodes to exactly one amateur (Qwen) token, and only align those. Tokens without a clean 1:1 match fall back to a multi-token subword sync path.

This mapping is necessarily partial — see Limitations.

Example outputs

Run with default settings (alpha=0.40, greedy decoding). Not cherry-picked further than "first run at these settings."

Prompt: "Who won the Nobel Prize in Physics in 1823?"

There was no 1823 Nobel Prize in Physics. The first Nobel Prizes were awarded in 1901.

Prompt: "What was the first domestic automobile brand produced in Türkiye in the 1970s, and what were its technical specifications?"

I was unable to verify the 1st domestic automobile brand, produced in 1970s, in Türkiye.

Both are trick questions (no 1823 physics Nobel exists; the specific claim is under-specified/unverifiable from the model's knowledge). The model declines to confabulate rather than inventing a plausible-sounding name — which is the behavior this technique is meant to encourage. We have not yet run a controlled comparison against the expert model alone on the same prompts — that comparison is the next thing on the list, and the results above should be read as anecdotal until it's done.

Known limitations

Being upfront about these matters more than pretending they don't exist:

English only, for now. Non-English prompts (tested: Turkish, German) currently produce corrupted output — words run together, subwords get mangled. Root cause: tokens are decoded and re-encoded one at a time during generation, and BPE tokenizers store leading-space information in a way that gets lost outside of full-sequence context. This effect is mild in English (where Llama/Qwen subword vocabularies overlap heavily) and severe in morphologically rich languages. Not yet fixed.
Greedy decoding only. No sampling, no temperature. Long generations can repeat despite a windowed repetition penalty; the penalty helps but isn't a substitute for proper sampling strategies.
Partial vocabulary mapping. Only tokens with a clean 1:1 string match between tokenizers get a real contrastive signal; everything else falls back to a slower multi-forward-pass sync path. The exact coverage percentage hasn't been measured yet.
No quantitative hallucination benchmark yet. Everything above is qualitative/anecdotal. A proper eval (e.g. against TruthfulQA or a similar factuality benchmark, expert-only vs fused) is the natural next step and isn't done.
Speed: ~7 tokens/sec on a free Colab T4 with 4-bit quantization, ~7.3 GB peak VRAM. Not fast; this is a research prototype, not a production serving setup.
Tech stack
PyTorch, Hugging Face Transformers
bitsandbytes (4-bit NF4 quantization), accelerate
meta-llama/Llama-3.2-3B-Instruct (gated — needs HF access approval) + Qwen/Qwen2.5-1.5B-Instruct
Running it

Colab: open the notebook, run all cells, authenticate with a Hugging Face token that has been granted access to Llama-3.2.

Local / VS Code:

bash
python -m venv .venv && source .venv/bin/activate
pip install transformers torch huggingface_hub bitsandbytes accelerate
huggingface-cli login
python si_engine.py --prompt "Who won the Nobel Prize in Physics in 1823?"
# or interactively:
python si_engine.py
Contributing

This is an early-stage research prototype, not a finished tool. If you want to help with any of the following, open an issue or PR:

A proper quantitative eval (expert-only vs fused, on a factuality benchmark)
Fixing multilingual generation (full-sequence re-decoding instead of per-token decode)
Measuring actual vocabulary mapping coverage between different tokenizer pairs
Testing other expert/amateur model pairs

License: MIT
