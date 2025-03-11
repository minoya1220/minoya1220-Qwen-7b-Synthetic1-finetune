import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_model(model_name="Qwen/Qwen-7B"):
    """Load and prepare Qwen model for training"""
    # Load tokenizer
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        model_max_length=2048,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded with vocabulary size: {tokenizer.vocab_size}")
    
    # Load model with appropriate precision
    print(f"Loading model from {model_name}...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # Disable KV cache for training
    model.config.use_cache = False
    
    # Test model with a sample input if CUDA is available
    if torch.cuda.is_available():
        print("Testing model with sample input...")
        sample_text = "Hello, I am a language model."
        sample_tokens = tokenizer(sample_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = model(input_ids=sample_tokens.input_ids, attention_mask=sample_tokens.attention_mask)
            inference_time = time.time() - start_time
            
            print(f"Forward pass successful in {inference_time:.2f}s")
            print(f"Output shape: {outputs.logits.shape}")
            print(f"Max GPU memory: {torch.cuda.max_memory_allocated(0) / (1024 ** 3):.2f} GB")
    
    return model, tokenizer