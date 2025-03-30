import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# Import the specific block class for Qwen models
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from typing import Optional, Tuple

def get_transformer_block_class(model):
    """
    Identifies and returns the specific transformer block class used by the model.
    This is needed for FSDP's auto_wrap_policy.
    """
    # For Qwen-7B (likely Qwen2 architecture), the block is typically Qwen2DecoderLayer
    # You might need to inspect the model architecture if using a different variant
    # print(model) # Uncomment this to print model structure if unsure
    return Qwen2DecoderLayer

def prepare_model(model_name="Qwen/Qwen-7B"):
    """Load and prepare Qwen model for training with FSDP support"""
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
    
    # Identify transformer blocks for FSDP wrapping (optional now, but good for verification)
    # For Qwen models, the transformer blocks are typically named "QWenBlock" or "Qwen2DecoderLayer"
    # This will be used by FSDP auto_wrap_policy
    identify_transformer_blocks(model) # Keep this call for verification/debugging
    
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

def identify_transformer_blocks(model):
    """Helper function to print transformer block names (for debugging)."""
    print("Inspecting model layers for transformer block class name:")
    found_blocks = set()
    for name, module in model.named_modules():
         # Check common patterns or specific class names
         if 'block' in name.lower() or 'layer' in name.lower():
             # Add the class name to the set to see unique types
             found_blocks.add(module.__class__.__name__)
    if found_blocks:
         print(f"Identified potential transformer block classes: {found_blocks}")
    else:
         print("Could not automatically identify transformer block names.")
