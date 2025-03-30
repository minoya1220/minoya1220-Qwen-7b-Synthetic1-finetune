import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling


def explore_dataset_structure():
    """Explore and print the structure of the SYNTHETIC-1 dataset in detail"""
    print("\n===== EXPLORING DATASET STRUCTURE =====")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
    
    # Print basic dataset info
    print(f"\nDataset splits: {list(dataset.keys())}")
    
    # Check train split
    train_split = dataset.get("train")
    if train_split is None:
        print("WARNING: 'train' split not found in the dataset!")
        return None
    elif len(train_split) == 0:
        print("WARNING: Train split is empty!")
    else:
        print(f"Number of examples in train split: {len(train_split)}")
        print("Train split seems okay.")

    return dataset


def prepare_dataset(tokenizer, max_length=2048, val_split=0.05):
    """Prepare the SYNTHETIC-1 dataset with validation split"""
    print("\nPreparing dataset...")
    
    # First load and explore the dataset structure
    dataset = explore_dataset_structure()
    if dataset is None or "train" not in dataset:
         raise ValueError("Failed to load or validate the initial dataset.")

    train_dataset = dataset["train"]

    # Split into train and validation
    print(f"Splitting train data with validation size: {val_split}")
    if len(train_dataset) == 0:
        raise ValueError("Cannot split an empty train dataset.")
        
    split_dataset = train_dataset.train_test_split(test_size=val_split, seed=42)
    print(f"Split into train ({len(split_dataset['train'])}) and validation ({len(split_dataset['test'])}) sets")
    
    # Format conversations
    print("\nFormatting conversations...")
    num_cpus = os.cpu_count()
    formatted_dataset = split_dataset.map( 
        format_conversation,
        num_proc=num_cpus,
        desc="Formatting conversations"
    )
        
    # Check if any examples have empty formatted text after formatting
    def check_empty(example):
        return len(example.get('formatted_text', '').strip()) == 0

    empty_train_count = formatted_dataset['train'].filter(check_empty).num_rows
    empty_test_count = formatted_dataset['test'].filter(check_empty).num_rows

    if empty_train_count > 0:
        print(f"WARNING: {empty_train_count} examples in train set resulted in empty formatted_text!")
    if empty_test_count > 0:
        print(f"WARNING: {empty_test_count} examples in test set resulted in empty formatted_text!")
        
    # Filter out empty examples if necessary, although tokenization might handle them
    # formatted_dataset = formatted_dataset.filter(lambda x: len(x.get('formatted_text', '').strip()) > 0)
    # print("Filtered out examples with empty formatted text.")


    # Tokenize the dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=formatted_dataset["train"].column_names,
        num_proc=num_cpus,
        desc="Tokenizing dataset"
    )
    
    # Final check for empty datasets after tokenization
    if len(tokenized_dataset['train']) == 0:
        raise ValueError("Train dataset is empty after tokenization! Check formatting, tokenization, or filtering steps.")
    if len(tokenized_dataset['test']) == 0:
        print("Warning: Test dataset is empty after tokenization.")

    print("Dataset preparation finished.")
    return tokenized_dataset


def format_conversation(example):
    """Convert the messages format to a simple text format"""
    formatted = ""
    messages = example.get("messages")
    if not isinstance(messages, list):
        print(f"Warning: Skipping example due to missing or invalid 'messages' field: {example}")
        return {"formatted_text": ""} 

    for msg in messages:
        if not isinstance(msg, dict):
            print(f"Warning: Skipping invalid message format (not a dict): {msg}")
            continue
        
        role = msg.get("role")
        content = msg.get("content")

        if not isinstance(role, str) or not isinstance(content, str):
             print(f"Warning: Skipping message with missing/invalid role or content: {msg}")
             continue
            
        if role == "user":
            formatted += f"USER: {content}\n\n"
        elif role == "assistant":
            formatted += f"ASSISTANT: {content}\n\n"

    return {"formatted_text": formatted.strip()}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the formatted text"""
    try:
        # Handle both single examples and batches
        texts = examples["formatted_text"]
        if not isinstance(texts, list):
            texts = [texts]
        
        # Ensure all texts are strings
        texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False
        )
    except Exception as e:
        print(f"Error tokenizing: {e}")
        batch_size = len(examples["formatted_text"]) if isinstance(examples["formatted_text"], list) else 1
        return {"input_ids": [[]] * batch_size, "attention_mask": [[]] * batch_size}


def create_data_collator(tokenizer, max_length=2048):
    """Create a data collator for causal language modeling"""
    print("\nCreating data collator using DataCollatorForLanguageModeling...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    print("Data collator created.")
    return data_collator


# If you want to run this file directly for exploration
if __name__ == "__main__":
    print("Running data exploration...")
    explore_dataset_structure()
    