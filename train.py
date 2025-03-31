import torch
import os
import gc
from transformers import Trainer, TrainingArguments, AutoModel
import wandb
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
import enum

from model import prepare_model, get_transformer_block_class
from data import prepare_dataset, create_data_collator
from utils import WandbCallback, print_device_info, init_wandb, create_output_dir

def create_fsdp_config(transformer_layer_cls):
    """Create FSDP configuration for training"""
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Define the transformer auto wrap policy using the provided class
    # Ensure transformer_layer_cls is the actual class object, not a string
    qwen_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ transformer_layer_cls, }
    )

    return {
        # "fsdp_transformer_layer_cls_to_wrap": "QWenBlock", # Replaced by policy below
        "fsdp_backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "fsdp_sharding_strategy": ShardingStrategy.FULL_SHARD,
        "fsdp_auto_wrap_policy": qwen_auto_wrap_policy, # Use transformer policy
        # "fsdp_min_num_params": 1e6, # Not needed for transformer_auto_wrap_policy
        "fsdp_state_dict_type": StateDictType.FULL_STATE_DICT,
        "fsdp_mixed_precision": bf16_policy if torch.cuda.is_available() else None,
        "fsdp_offload_params": False,
        "fsdp_use_orig_params": True, # Recommended for HF Trainer compatibility
    }

def create_training_args(output_dir, num_epochs, batch_size, gradient_accumulation_steps, 
                        learning_rate, fsdp_config, use_wandb):
    """Create training arguments"""
    return TrainingArguments(
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
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        fsdp=fsdp_config,
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

def train(
    model_name="Qwen/Qwen-7B",
    output_dir=None,
    batch_size=48,
    gradient_accumulation_steps=4,
    num_epochs=1,
    learning_rate=2e-5,
    max_length=2048,
    val_split=0.05,
):
    """Main training function optimized for 4x H100s"""
    # Setup
    print_device_info()
    output_dir = create_output_dir(output_dir)
    use_wandb = init_wandb(model_name, batch_size, learning_rate, num_epochs, 
                         max_length, gradient_accumulation_steps)
    
    try:
        # Load model, tokenizer, AND the transformer block class
        # NOTE: prepare_model in model.py needs modification to return the class
        model, tokenizer = prepare_model(model_name)
        # Get the specific transformer block class for FSDP wrapping
        # This function needs to be implemented in model.py
        transformer_block_class = get_transformer_block_class(model) 
        if transformer_block_class is None:
             # Fallback or error if class not found
             print("Warning: Could not identify transformer block class. FSDP might not be optimal.")
             # Decide fallback strategy: maybe use size_based policy or raise error
             # For now, let's assume it will be found for Qwen models
             # You might need to import the class directly if get_transformer_block_class fails
             # e.g., from transformers.models.qwen.modeling_qwen import QWenBlock
             # transformer_block_class = QWenBlock 

        dataset = prepare_dataset(tokenizer, max_length, val_split)
        # Remove max_length from data_collator call
        data_collator = create_data_collator(tokenizer) 
        
        # Configure training with FSDP, passing the block class
        fsdp_config = create_fsdp_config(transformer_block_class) # Pass the class object

        if fsdp_config and isinstance(fsdp_config, dict):
            # Create a new dict to avoid modifying the original if it's used elsewhere
            serializable_fsdp_config = {}
            for key, value in fsdp_config.items():
                # Check specifically for FSDP enums or general enums
                if isinstance(value, (BackwardPrefetch, ShardingStrategy, StateDictType, MixedPrecision, enum.Enum)):
                    if isinstance(value, enum.Enum):
                         serializable_fsdp_config[key] = value.name
                    else:
                         
                         serializable_fsdp_config[key] = str(value) # Fallback to string representation

                # Handle the auto_wrap_policy (functools.partial) - Not directly JSON serializable
                elif key == "fsdp_auto_wrap_policy":
                     # maybe delete 
                     serializable_fsdp_config[key] = f"functools.partial(<{value.func.__name__}>, ...)"
                     print("Warning: fsdp_auto_wrap_policy converted to string for logging.")
                else:
                    serializable_fsdp_config[key] = value
        else:
             serializable_fsdp_config = fsdp_config # Pass original if not a dict or None

        training_args = create_training_args(output_dir, num_epochs, batch_size,
                                          gradient_accumulation_steps, learning_rate,
                                          serializable_fsdp_config, use_wandb) # <-- Use the serializable version
        
        # Setup callbacks
        callbacks = [WandbCallback()] if use_wandb else []
        
        # Initialize trainer
        print("DEBUG: Initializing Trainer object...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            callbacks=callbacks
        )
        print("DEBUG: Trainer object initialized.")
        
        # Train and evaluate
        print("\nStarting training...")
        train_result = trainer.train()
        print("DEBUG: trainer.train() finished.")
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Run final evaluation
        print("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        # Clean up
        if use_wandb:
            wandb.finish()
        
        # Free memory
        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if use_wandb:
            wandb.finish()
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return False
    