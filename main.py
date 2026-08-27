# ==============================================================================
# PROJECT: SI (Synthetic Intelligence Architecture)
# SUB-SYSTEM: Pure String-Based Production Stopping Framework (Bug-Free)
# INFRASTRUCTURE: PyTorch, Transformers, BitsAndBytes, Accelerate
# PLATFORM: Local Machine / Server / Cloud VM (CUDA Supported)
# ==============================================================================

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("==================================================================")
print("🧠 SI (SYNTHETIC INTELLIGENCE) CORE UTILITIES INITIALIZED")
print("==================================================================")

# 1. API Access Control via Environment Variable
# Sisteminizde HF_TOKEN adında bir çevre değişkeni yoksa terminalden elle istenir.
if "HF_TOKEN" not in os.environ:
    print("\n🔒 [SI SECURITY] Please input your Hugging Face Access Token:")
    hf_token = input("Token: ").strip()
    os.environ["HF_TOKEN"] = hf_token

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 [SI ENGINE] Active Hardware Layer: {device.upper()}")

# Hardware Optimization: 4-Bit Model Compression Framework
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 2. Model Orchestration & Synchronization Pipeline
model_expert_name = "meta-llama/Llama-3.2-3B-Instruct"  
model_amateur_name = "Qwen/Qwen2.5-1.5B-Instruct"       

print("\n⏳ Downloading and loading dual neural structures into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(model_expert_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Modeller otomatik olarak os.environ["HF_TOKEN"] değerini kullanacaktır.
model_expert = AutoModelForCausalLM.from_pretrained(model_expert_name, quantization_config=quant_config_4bit)
model_amateur = AutoModelForCausalLM.from_pretrained(model_amateur_name, quantization_config=quant_config_4bit)

print("\n🎯 [SI ENGINE] Cross-Cultural Neurons Successfully Linked!")
print("==================================================================")
print("💬 SI INTERACTIVE PRODUCTION CONSOLE")
print("Type 'exit' or 'quit' to terminate the session safely.")
print("==================================================================")

# 3. Global Interactive Console Production Loop
while True:
    user_query = input("\n👤 User: ")
    
    if user_query.lower() in ['exit', 'quit']:
        print("\n🤖 SI Engine: Hardware session closed. Goodbye!")
        break
        
    if not user_query.strip():
        continue
        
    messages = [{"role": "user", "content": user_query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    print("🤖 SI: ", end="", flush=True)

    alpha = 0.35             
    generated_tokens = []
    soft_limit = 100         
    absolute_max = 180       

    for token_index in range(absolute_max):
        with torch.no_grad():
            outputs_expert = model_expert(input_ids)
            outputs_amateur = model_amateur(input_ids)
            
            logits_expert = outputs_expert.logits[:, -1, :]
            logits_amateur = outputs_amateur.logits[:, -1, :]
            
            if logits_expert.shape[-1] != logits_amateur.shape[-1]:
                min_vocab_size = min(logits_expert.shape[-1], logits_amateur.shape[-1])
                logits_expert = logits_expert[:, :min_vocab_size]
                logits_amateur = logits_amateur[:, :min_vocab_size]
            
            fused_logits = logits_expert - (alpha * logits_amateur)
            fused_logits = fused_logits / 0.7
            
            if token_index >= (absolute_max - 5):
                fused_logits[0, tokenizer.eos_token_id] += 100.0

            next_token = torch.argmax(fused_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_tokens.append(next_token.item())
            
            current_word = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            
            if token_index >= soft_limit:
                if any(punct in current_word for punct in [".", "!", "?"]):
                    break
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    clean_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(clean_output)
    print("-" * 60)
