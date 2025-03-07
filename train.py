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
    return {
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
        "bf16": {"enabled": torch.cuda.is_available()},
        "gradient_clipping": 1.0,
        "train_batch_size": batch_size * torch.cuda.device_count() * gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 50,
        "wall_clock_breakdown": False
    }

def create_training_args(output_dir, num_epochs, batch_size, gradient_accumulation_steps, 
                        learning_rate, ds_config, use_wandb):
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
        # Load model and prepare dataset
        model, tokenizer = prepare_model(model_name)
        dataset = prepare_dataset(tokenizer, max_length, val_split)
        data_collator = create_data_collator(tokenizer, max_length)
        
        # Configure training
        ds_config = create_deepspeed_config(batch_size, gradient_accumulation_steps)
        training_args = create_training_args(output_dir, num_epochs, batch_size,
                                          gradient_accumulation_steps, learning_rate,
                                          ds_config, use_wandb)
        
        # Setup callbacks
        callbacks = [WandbCallback()] if use_wandb else []
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train and evaluate
        print("\nStarting training...")
        train_result = trainer.train()
        
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