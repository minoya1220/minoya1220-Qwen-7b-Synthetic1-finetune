import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_model(model_name="Qwen/Qwen-7B"):
    """Load and prepare Qwen model for training"""
    print(f"Loading tokenizer from {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            model_max_length=2048,
            padding_side="left",  # Better for causal LM training
        )
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
        print(f"Tokenizer loaded successfully: {tokenizer is not None}")
        print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise  # Critical failure in tokenizer loading
    
    # Initialize model with mixed precision
    print(f"Loading model from {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Prepare model for training
        model.config.use_cache = False
        print(f"Model loaded successfully: {model is not None}")
        
        # Test model with a small input if CUDA is available
        if torch.cuda.is_available():
            print("Testing model with sample input...")
            sample_text = "Hello, I am a language model."
            sample_tokens = tokenizer(sample_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_ids=sample_tokens.input_ids, attention_mask=sample_tokens.attention_mask)
                print(f"Forward pass successful. Time: {time.time() - start_time:.2f}s")
                print(f"Output shape: {outputs.logits.shape}")
                
                # Report GPU memory usage
                max_memory = torch.cuda.max_memory_allocated(0) / (1024 ** 3)  # GB
                print(f"Max GPU memory allocated: {max_memory:.2f} GB")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise  # Critical failure in model loading
        
    return model, tokenizer