import torch
import os
import gc
from transformers import Trainer, TrainingArguments
import wandb

from model import prepare_model
from data import prepare_dataset, create_data_collator
from utils import WandbCallback, print_device_info, init_wandb, create_output_dir

def create_deepspeed_config(batch_size, gradient_accumulation_steps):
    """Create DeepSpeed ZeRO-3 configuration"""
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
    return ds_config

def create_training_args(output_dir, num_epochs, batch_size, gradient_accumulation_steps, 
                        learning_rate, ds_config, use_wandb):
    """Create training arguments"""
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
    return training_args

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
    print_device_info()
    
    # Create output directory
    output_dir = create_output_dir(output_dir)
    
    # Initialize wandb
    use_wandb = init_wandb(model_name, batch_size, learning_rate, num_epochs, 
                         max_length, gradient_accumulation_steps)
    
    try:
        # Prepare model and tokenizer
        model, tokenizer = prepare_model(model_name)
        
        # Prepare dataset with validation split
        datasets = prepare_dataset(tokenizer, max_length, val_split)
        
        # Create data collator with dynamic padding
        data_collator = create_data_collator(tokenizer, max_length)
        
        # DeepSpeed ZeRO-3 config optimized for H100s
        ds_config = create_deepspeed_config(batch_size, gradient_accumulation_steps)
        
        # Training arguments optimized for H100s
        training_args = create_training_args(
            output_dir, num_epochs, batch_size, gradient_accumulation_steps,
            learning_rate, ds_config, use_wandb
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