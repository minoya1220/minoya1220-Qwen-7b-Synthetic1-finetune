import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training
import deepspeed

def prepare_model(model_name="Qwen/Qwen-7B"):
    """Load and prepare Qwen model for training"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        model_max_length=2048,
    )
    
    # Initialize model with mixed precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Prepare model for training
    model.config.use_cache = False
    return model, tokenizer

def prepare_dataset(tokenizer, max_length=2048):
    """Prepare the SYNTHETIC-1 dataset"""
    dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
    
    def format_conversation(item):
        # Format into chat format
        messages = item['messages']
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            formatted += f"{role}: {content}\n"
        return formatted
    
    def tokenize_function(examples):
        texts = [format_conversation(x) for x in examples['messages']]
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
    
    # Process dataset in parallel
    tokenized_dataset = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        num_proc=16  # Increased for H100s
    )
    
    return tokenized_dataset

def train(
    model_name="Qwen/Qwen-7B",
    output_dir="./synthetic1_output",
    batch_size=48,  # Adjusted for H100s (80GB)
    gradient_accumulation_steps=1,
    num_epochs=1,
    learning_rate=2e-5,
    max_length=2048,
):
    """Main training function optimized for 4x H100s"""
    model, tokenizer = prepare_model(model_name)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer, max_length)
    
    # DeepSpeed ZeRO-3 config optimized for H100s
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
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "train_batch_size": batch_size * 4,  # Total across 4 GPUs
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 50,
        "wall_clock_breakdown": False
    }
    
    # Training arguments optimized for H100s
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=ds_config,
        report_to="tensorboard",
        # H100-specific optimizations
        dataloader_num_workers=8,  # Increased for H100s
        dataloader_pin_memory=True,
        group_by_length=True,
        length_column_name="length",
        ignore_data_skip=True,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data])
        }
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train(
        model_name="Qwen/Qwen-7B",
        output_dir="./synthetic1_output"
    )