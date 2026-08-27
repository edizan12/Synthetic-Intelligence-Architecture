# ==============================================================================
# PROJECT: SI (Synthetic Intelligence Architecture)
# SUB-SYSTEM: Pure String-Based Production Stopping Framework (Bug-Free)
# INFRASTRUCTURE: PyTorch, Transformers, BitsAndBytes, Accelerate
# PLATFORM: Google Colab - T4 GPU (16GB VRAM)
# ==============================================================================

# 1. Install global enterprise libraries
!pip install -q transformers torch huggingface_hub bitsandbytes accelerate

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import notebook_login

print("==================================================================")
print("🧠 SI (SYNTHETIC INTELLIGENCE) CORE UTILITIES INITIALIZED")
print("==================================================================")

# 2. Secure Gated Authentication & API Access Control
# This triggers the login widget asking for the user's hf_... token dynamically
print("\n🔒 [SI SECURITY] Please input your Hugging Face Access Token:")
notebook_login()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 [SI ENGINE] Active Hardware Layer: {device.upper()}")

# Hardware Optimization: 4-Bit Model Compression Framework
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 3. Model Orchestration & Synchronization Pipeline
# Pulls weights from Western (Llama) and Eastern (Qwen) open-source repositories
model_expert_name = "meta-llama/Llama-3.2-3B-Instruct"  
model_amateur_name = "Qwen/Qwen2.5-1.5B-Instruct"       

print("\n⏳ Downloading and loading dual neural structures into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(model_expert_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Automatically downloads models securely using the authenticated HF API token
model_expert = AutoModelForCausalLM.from_pretrained(model_expert_name, quantization_config=quant_config_4bit)
model_amateur = AutoModelForCausalLM.from_pretrained(model_amateur_name, quantization_config=quant_config_4bit)

print("\n🎯 [SI ENGINE] Cross-Cultural Neurons Successfully Linked!")
print("==================================================================")
print("💬 SI INTERACTIVE PRODUCTION CONSOLE")
print("Type 'exit' or 'quit' to terminate the session safely.")
print("==================================================================")

# 4. Global Interactive Console Production Loop
while True:
    # Capture live user query from the console
    user_query = input("\n👤 User: ")
    
    # Secure escape sequence to terminate the hardware session gracefully
    if user_query.lower() in ['exit', 'quit']:
        print("\n🤖 SI Engine: Hardware session closed. Goodbye!")
        break
        
    if not user_query.strip():
        continue
        
    # Apply formal chat template tokenization for unified input pipeline
    messages = [{"role": "user", "content": user_query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    print("🤖 SI: ", end="", flush=True)

    alpha = 0.35             # Contrastive decoding penalty coefficient
    generated_tokens = []
    soft_limit = 100         # Token threshold where the engine starts looking for a safe exit
    absolute_max = 180       # Hardware ceiling limit to prevent VRAM overflow

    for token_index in range(absolute_max):
        with torch.no_grad():
            # Concurrently extract raw hidden layers from both expert and amateur architectures
            outputs_expert = model_expert(input_ids)
            outputs_amateur = model_amateur(input_ids)
            
            logits_expert = outputs_expert.logits[:, -1, :]
            logits_amateur = outputs_amateur.logits[:, -1, :]
            
            # Synchronize tensor dimensions to eliminate heterogeneous vocabulary size anomalies
            if logits_expert.shape[-1] != logits_amateur.shape[-1]:
                min_vocab_size = min(logits_expert.shape[-1], logits_amateur.shape[-1])
                logits_expert = logits_expert[:, :min_vocab_size]
                logits_amateur = logits_amateur[:, :min_vocab_size]
            
            # Pure Mathematical Contrastive Decoding Layer
            fused_logits = logits_expert - (alpha * logits_amateur)
            fused_logits = fused_logits / 0.7
            
            # Hard boundary enforcement: Bias logits towards EOS token when approaching absolute ceiling
            if token_index >= (absolute_max - 5):
                fused_logits[0, tokenizer.eos_token_id] += 100.0

            next_token = torch.argmax(fused_logits, dim=-1, keepdim=True)
            
            # Append selected token tensor to active attention sequence and local tracking array
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())
            
            # === ENTERPRISE LOGIC: REAL-TIME STRING PATTERN SAMPLING ===
            # Instantly decode the single predicted token into an active string chunk
            current_word = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            
            # Soft-stopping layer: Terminate generation seamlessly if a natural punctuation boundary is found after soft_limit
            if token_index >= soft_limit:
                if any(punct in current_word for punct in [".", "!", "?"]):
                    break
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # Output post-processing and text cleaning layer
    clean_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(clean_output)
    print("-" * 60)
