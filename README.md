# SI: Cross-Tokenizer Contrastive Decoding

Combine two LLMs with different tokenizers at inference time to steer generation toward probability mass that is distinctive to the expert model — no fine-tuning, no external judge or agent calls.

This is an experimental research prototype for inference-time contrastive decoding with cross-tokenizer alignment.

## The idea

Most hallucination-mitigation setups use a text-level multi-agent pattern: one model generates an answer, another model reads and critiques it through a second call. That approach operates on decoded text.

SI instead operates directly on the models' token probability distributions during generation:

```text
prompt
  │
  ├──► Llama-3.2-3B (expert) ──► logits_E
  │
  └──► Qwen2.5-1.5B (amateur) ──► logits_A
                         │
                         ▼
                log-softmax distributions
                         │
                         ▼
             fused = lp_E − α · lp_A
                         │
                         ▼
               β plausibility filter
                         │
                         ▼
             repetition / EOS handling
                         │
                         ▼
                      next token
```

The intuition is related to contrastive decoding (Li et al., 2022) and DoLa-style methods: tokens that are also highly probable under a smaller model can represent generic or less distinctive completions. Subtracting a scaled amateur distribution can therefore shift generation toward probability mass that is more distinctive to the expert model.

This is a decoding hypothesis, not a claim that SI universally improves factuality or hallucination resistance.

## Why cross-tokenizer alignment is the hard part

Llama and Qwen do not share the same vocabulary. Comparing their logit tensors index-by-index would therefore be meaningless.

SI builds a string-based mapping between the two tokenizers. For each expert token, it checks whether its decoded string can be re-encoded by the amateur tokenizer as a clean one-to-one token. Tokens without a clean 1:1 match use a proxy/subtoken synchronization path.

For the current Llama/Qwen pair, the mapping has been measured as:

- Expert vocabulary: **128,256 tokens**
- Clean 1:1 mappings: **109,778**
- Proxy mappings: **18,218**
- Total mapped coverage: **127,996 / 128,256 = 99.8%**
- Unmapped expert tokens: **260**

The mapping is therefore highly overlapping for this model/tokenizer pair, but it is not complete.

## The beta plausibility constraint

Raw contrastive fusion can become unstable as `alpha` increases.

The current implementation therefore applies a plausibility constraint after fusion:

```python
fused = lp_e - alpha * mapped_am

threshold = lp_e.max() + math.log(beta)

fused = fused.masked_fill(
    lp_e < threshold,
    -float("inf")
)
```

The purpose is to prevent an amateur-model contrastive bonus from promoting a token that the expert model itself considers extremely implausible.

In a fixed single-step ablation without the beta constraint, increasing `alpha` caused low-expert-probability tokens to become top candidates under raw fusion. The current generation path keeps the beta plausibility filter enabled to suppress this failure mode.

This does **not** prove that beta improves factual accuracy. It demonstrates that beta is an important stability constraint for the raw contrastive objective.

## Current decoding implementation

The current v9.5.4 implementation:

- maintains KV caches for both models during multi-step generation
- performs log-probability fusion in float32
- supports beta plausibility filtering
- supports repetition penalties with a configurable window
- includes explicit EOS handling near the generation limit
- supports greedy decoding at `temperature=0`
- supports sampling and optional top-p filtering at non-zero temperature
- synchronizes the amateur model through direct 1:1 token mapping or the proxy/subtoken path

## Validation status

The implementation has been exercised with unit-style tests covering:

- vocabulary mapping
- proxy mapping and padding
- dtype regression
- `alpha = 0` behavior
- unmapped-token invariance
- beta plausibility filtering and boundary behavior
- repetition penalties
- EOS forcing
- mock end-to-end generation

Real-model generation has also been tested with Llama-3.2-3B-Instruct and Qwen2.5-1.5B-Instruct on CUDA.

These tests establish implementation behavior and expose stability/failure modes. They do **not** establish that SI produces fewer hallucinations than expert-only decoding.

## Qualitative factuality observations

Early outputs provide qualitative examples, but they are not a controlled factuality evaluation.

For example, one generation correctly declined an impossible historical premise such as a Nobel Prize in Physics in 1823. Other outputs have also shown factual imprecision. At `alpha=0.4`, one answer defining "infodemic" described it too generally as the widespread transmission of information, omitting the important association with an overabundance of information that can include false or misleading information.

This is a qualitative observation, not evidence that the error was caused specifically by contrastive fusion. Controlled expert-only versus SI comparisons and token-level attribution are still needed to determine how `alpha` affects factual accuracy.

## Known limitations

### English is currently the primary supported case

Turkish generation has produced corrupted or awkward subword output in testing. The likely mechanism is related to decoding and re-encoding individual tokens outside full-sequence tokenizer context, particularly around leading-space and subword behavior.

This has not yet been fully fixed.

### Contrastive decoding sensitivity

The choice of `alpha` matters.

Without the beta plausibility constraint, larger `alpha` values can produce pathological token selections. There is not yet a principled, benchmark-derived `alpha` / `beta` setting that is known to work optimally across prompts or model pairs.

### Partial vocabulary mapping

The current model pair achieves **99.8% mapped coverage**, but the remaining expert tokens do not have clean one-to-one alignment.

The proxy/subtoken fallback path is slower and more complicated than direct 1:1 mapping.

### No quantitative hallucination benchmark yet

There is currently no controlled benchmark comparing expert-only and SI decoding on a factuality or hallucination dataset.

A proper evaluation should compare at least:

- expert-only decoding
- SI decoding
- multiple `alpha` values
- multiple `beta` values

on the same prompts and evaluation set.

### Performance

This is a research prototype rather than a production serving system.

Observed throughput has been in the mid-single-digit tokens/sec range in real-model runs; one 134-token generation at `alpha=0.4`, `beta=0.1`, `rep=0.3`, `T=0.0` measured **7.28 tokens/sec**.

GPU memory usage ranged roughly from **3.6 GB to 6.0 GB** across different runs in this development session. The cause of this variation has not been isolated in a controlled benchmark.

These are development-session observations, not controlled throughput or VRAM benchmarks.

## Tech stack

- PyTorch
- Hugging Face Transformers
- bitsandbytes (4-bit NF4 quantization)
- accelerate

Models:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`

The Llama model is gated and requires appropriate Hugging Face access.

## Running it

### Colab

Open the notebook, install the dependencies, and authenticate with a Hugging Face token that has access to the Llama model.

### Local / VS Code

```bash
python -m venv .venv
source .venv/bin/activate

pip install transformers torch huggingface_hub bitsandbytes accelerate

huggingface-cli login

python main.py --prompt "Who won the Nobel Prize in Physics in 1823?"
```

Or interactively:

```bash
python main.py
```

## Contributing

This is an early-stage research prototype, not a finished tool.

Useful next steps include:

- controlled expert-only vs SI factuality evaluation
- systematic `alpha` / `beta` sweeps
- token-level generation tracing and attribution
- fixing multilingual generation
- improving full-sequence tokenizer synchronization
- testing additional expert/amateur model pairs
- systematic throughput and VRAM benchmarking

## License

MIT
