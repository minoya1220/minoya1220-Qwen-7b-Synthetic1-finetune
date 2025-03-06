import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
import deepspeed
import wandb
import os
import gc
import time
from datetime import datetime
from tqdm import tqdm

# Add a WandB callback to log metrics
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

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

def prepare_dataset(tokenizer, max_length=2048, val_split=0.05):
    """Prepare the SYNTHETIC-1 dataset with validation split"""
    print("\nPreparing dataset...")
    try:
        print("Attempting to load dataset...")
        dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
        print(f"Dataset loaded successfully: {dataset is not None}")
        print(f"Dataset keys: {dataset.keys()}")
        if 'train' in dataset:
            print(f"Train split size: {len(dataset['train'])}")
            print(f"Sample record structure: {list(dataset['train'][0].keys())}")
            print(f"First example messages count: {len(dataset['train'][0]['messages'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise  # Critical failure in dataset loading
    
    # Split into train and validation
    print("\nSplitting dataset into train and validation...")
    dataset = dataset["train"].train_test_split(test_size=val_split, seed=42)
    print(f"After split - Train: {len(dataset['train'])}, Validation: {len(dataset['test'])}")
    
    def format_conversation(example):
        # Format into chat format 
        # Note: Assuming each example has a 'messages' field with a list of message objects
        messages = example['messages']
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"{role}: {content}\n"
        return {"formatted_text": formatted}
    
    # Apply formatting to prepare chat format
    print("\nFormatting conversations...")
    try:
        formatted_dataset = dataset.map(
            format_conversation,
            num_proc=16,
            desc="Formatting conversations"
        )
        print("Formatting complete.")
        
        # Test a sample formatted conversation
        print("\nSample formatted conversation:")
        sample_idx = 0
        print(formatted_dataset['train'][sample_idx]['formatted_text'][:500] + "...")
    except Exception as e:
        print(f"Error formatting dataset: {e}")
        raise
    
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_text"],
            max_length=max_length,
            truncation=True,
            padding=False  # We'll use dynamic padding in the data collator
        )
    
    # Process dataset in parallel
    print("\nTokenizing dataset...")
    try:
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset["train"].column_names,
            num_proc=16,
            desc="Tokenizing dataset"
        )
        
        # Verify tokenization
        print("\nVerifying tokenization...")
        if len(tokenized_dataset['train']) > 0:
            sample = tokenized_dataset['train'][0]
            print(f"Sample tokenized shape - Input IDs: {len(sample['input_ids'])}, Attention Mask: {len(sample['attention_mask'])}")
            
            # Check for extremely long sequences
            lengths = [len(sample['input_ids']) for sample in tokenized_dataset['train'][:100]]
            avg_length = sum(lengths) / len(lengths)
            max_length_found = max(lengths)
            print(f"Average sequence length (first 100 samples): {avg_length:.1f}")
            print(f"Maximum sequence length (first 100 samples): {max_length_found}")
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        raise
    
    print(f"Train dataset size: {len(tokenized_dataset['train'])}")
    print(f"Validation dataset size: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset

def create_data_collator(tokenizer, max_length=2048):
    """Create a data collator with dynamic padding"""
    print("\nCreating data collator...")
    
    def collate_fn(examples):
        try:
            # Tokenize and pad dynamically in batch
            batch = {}
            batch["input_ids"] = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
            batch["attention_mask"] = torch.stack([torch.tensor(example["attention_mask"]) for example in examples])
            batch["labels"] = batch["input_ids"].clone()
            return batch
        except Exception as e:
            print(f"Error in collate function: {e}")
            # Return a default batch structure if possible
            raise
    
    # Test the collator with a small batch
    try:
        print("Testing data collator with sample batch...")
        sample_batch = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [5, 6, 7, 8], "attention_mask": [1, 1, 1, 1]},
        ]
        test_batch = collate_fn(sample_batch)
        print(f"Data collator test successful. Batch shape: {test_batch['input_ids'].shape}")
    except Exception as e:
        print(f"Warning: Data collator test failed: {e}")
        print("This could indicate issues during training.")
    
    return collate_fn

def train(
    model_name="Qwen/Qwen-7B",
    output_dir=None,
    batch_size=48,  # Adjusted for H100s (80GB)
    gradient_accumulation_steps=4,  # Increased for larger effective batch size
    num_epochs=1,
    learning_rate=2e-5,
    max_length=2048,
    val_split=0.05,
):
    """Main training function optimized for 4x H100s"""
    # Print system information
    print("=== Training Setup Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Create a timestamped output directory if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./synthetic1_output_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize wandb with more configuration options
    run_name = f"qwen-synthetic1-4xh100-{timestamp}"
    try:
        wandb.init(
            project="qwen-synthetic1", 
            name=run_name,
            config={
                "model": model_name,
                "batch_size": batch_size * torch.cuda.device_count() * gradient_accumulation_steps,  # Total effective batch size
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "max_length": max_length,
            }
        )
        print(f"WandB initialized with run name: {run_name}")
        use_wandb = True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Training will continue without WandB logging.")
        use_wandb = False
    
    try:
        model, tokenizer = prepare_model(model_name)
        
        # Prepare dataset with validation split
        datasets = prepare_dataset(tokenizer, max_length, val_split)
        
        # Create data collator with dynamic padding
        data_collator = create_data_collator(tokenizer, max_length)
        
        # DeepSpeed ZeRO-3 config optimized for H100s
        print("\nCreating DeepSpeed configuration...")
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e6,
                "gather_16bit_weights_on_model_save": True,
                "round_robin_gradients": True
            },
            "bf16": {
                "enabled": torch.cuda.is_available()
            },
            "gradient_clipping": 1.0,
            "train_batch_size": batch_size * torch.cuda.device_count() * gradient_accumulation_steps,  # Total across all GPUs
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 50,
            "wall_clock_breakdown": False
        }
        
        # Training arguments optimized for H100s
        print("\nCreating training arguments...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,  # Keep more checkpoints
            evaluation_strategy="steps",  # Add evaluation during training
            eval_steps=500,  # Evaluate every 500 steps
            bf16=torch.cuda.is_available(),  # Use bf16 if CUDA is available
            gradient_checkpointing=True,
            deepspeed=ds_config,
            report_to=["tensorboard", "wandb"] if use_wandb else ["tensorboard"],
            # H100-specific optimizations
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            group_by_length=True,
            length_column_name="length",
            ignore_data_skip=True,
            ddp_find_unused_parameters=False,
            # Add early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        callbacks = []
        # Create our custom wandb callback
        if use_wandb:
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)
        
        # Initialize trainer
        print("\nInitializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],  # Add validation dataset
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train
        print("\n=== Starting Training ===")
        print(f"Training on {len(datasets['train'])} examples")
        print(f"Validating on {len(datasets['test'])} examples")
        print(f"Using batch size of {batch_size} per GPU")
        print(f"Using {gradient_accumulation_steps} gradient accumulation steps")
        print(f"Effective batch size: {batch_size * torch.cuda.device_count() * gradient_accumulation_steps}")
        
        try:
            train_result = trainer.train()
            print("\n=== Training Complete ===")
            print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
            print(f"Training samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated(0) / (1024 ** 3)  # GB
                print(f"Max GPU memory allocated: {max_memory:.2f} GB")
        except Exception as e:
            print(f"\n=== Training Failed ===")
            print(f"Error: {e}")
            # Save checkpoint even if training fails
            print("Saving interrupted checkpoint...")
            try:
                trainer.save_model(os.path.join(output_dir, "interrupted_checkpoint"))
                print("Interrupted checkpoint saved successfully.")
            except Exception as save_error:
                print(f"Error saving interrupted checkpoint: {save_error}")
            raise
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Clean up to release memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Close wandb run
        if use_wandb:
            wandb.finish()
            
        print(f"\nTraining completed successfully. Model saved to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Critical error during setup: {e}")
        if use_wandb:
            wandb.finish()
        return False

if __name__ == "__main__":
    import argparse
    import sys
    
    print("\n=== Qwen-7B Synthetic1 SFT Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    parser = argparse.ArgumentParser(description="Train Qwen model on SYNTHETIC-1 dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Model name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=float, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Print arguments
    print("\nTraining Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    try:
        success = train(
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            val_split=args.val_split
        )
        
        if success:
            print("\n=== Training completed successfully! ===")
            sys.exit(0)
        else:
            print("\n=== Training failed. See error messages above. ===")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n=== Training interrupted by user ===")
        sys.exit(1)
    except Exception as e:
        print(f"\n=== Unexpected error: {e} ===")
        import traceback
        traceback.print_exc()
        sys.exit(1)