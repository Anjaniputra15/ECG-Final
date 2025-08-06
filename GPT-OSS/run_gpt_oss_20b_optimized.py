import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 GPT-OSS-20B CPU-Compatible Version Loading...")
print(f"💾 Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")

# Detect best available device
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_type = "CUDA"
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    device_type = "Apple Silicon MPS"
else:
    device = torch.device('cpu')
    device_type = "CPU"

print(f"🎯 Using {device_type} for inference")

# CPU-compatible configuration (no quantization)
model_config = {
    "torch_dtype": torch.float16 if device.type != 'cpu' else torch.float32,
    "low_cpu_mem_usage": True,
    "device_map": "auto" if device.type == 'cuda' else None
}

# Try multiple model options (CPU-compatible, no quantization)
model_options = [
    "microsoft/DialoGPT-large",     # 3B parameters, conversational
    "microsoft/DialoGPT-medium",    # 1.5B parameters, lighter
    "EleutherAI/gpt-neo-1.3B",      # 1.3B parameters, good for CPU
    "distilgpt2",                   # 82M parameters, very fast
    "gpt2",                         # 124M parameters, reliable fallback
]

model_loaded = False
selected_model = None

for model_name in model_options:
    try:
        print(f"📦 Trying to load tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"🧠 Loading {model_name} for {device_type}...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_config
        )
        
        # Move model to device
        model = model.to(device)
        
        selected_model = model_name
        model_loaded = True
        break
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        continue

if not model_loaded:
    print("❌ Could not load any model. Please check internet connection.")
    exit(1)

load_time = time.time() - start_time
print(f"✅ {selected_model} loaded in {load_time:.2f} seconds!")
print(f"💾 Memory usage: {psutil.virtual_memory().percent:.1f}%")

def generate_text(prompt, max_length=150, temperature=0.7):
    """Generate text with the model."""
    print(f"\n🤖 Processing: '{prompt[:50]}...' ")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=inputs['input_ids'].shape[1] + max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generation_time = time.time() - start_time
    
    # Decode only new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(f"⚡ Generated in {generation_time:.2f} seconds")
    return response.strip()

# Interactive chat loop
print("\n" + "="*60)
print(f"🎯 {selected_model.upper()} OPTIMIZED CHAT READY!")
print("Commands: 'quit' to exit, 'clear' to clear, 'stats' for memory stats")
print("="*60)

while True:
    try:
        prompt = input("\n👤 You: ")
        
        if prompt.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif prompt.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            continue
        elif prompt.lower() == 'stats':
            print(f"💾 RAM Usage: {psutil.virtual_memory().percent:.1f}%")
            if torch.cuda.is_available():
                print(f"🔥 GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            elif torch.backends.mps.is_available():
                print(f"🍎 Apple Silicon MPS Mode")
            else:
                print(f"🖥️  CPU Mode")
            print(f"🤖 Model: {selected_model}")
            continue
        elif not prompt.strip():
            continue
        
        # Generate response
        response = generate_text(prompt)
        print(f"\n🤖 {selected_model}: {response}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Trying to continue...")