import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm

def explore_dataset_structure():
    """Explore and print the structure of the SYNTHETIC-1 dataset in detail"""
    print("\n===== EXPLORING DATASET STRUCTURE =====")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
    
    # Print basic dataset info
    print(f"\nDataset splits: {list(dataset.keys())}")
    print(f"Number of examples in train split: {len(dataset['train'])}")
    
    # Get a sample example
    if len(dataset['train']) > 0:
        example = dataset['train'][0]
        print("\n----- Sample Example Structure -----")
        print(f"Top-level keys: {list(example.keys())}")
        
        # Debug: Check if 'messages' exists and its format
        if 'messages' in example:
            messages = example['messages']
            print(f"Messages field type: {type(messages)}")
            print(f"Messages count: {len(messages)}")
            
            # Print first message structure
            if len(messages) > 0:
                print("\nFirst message structure:")
                first_msg = messages[0]
                print(f"Message keys: {list(first_msg.keys())}")
                print(f"Role: {first_msg.get('role', 'unknown')}")
                content = first_msg.get('content', '')
                print(f"Content preview: {content[:100]}...")
    else:
        print("WARNING: Train split is empty!")
    
    return dataset


def prepare_dataset(tokenizer, max_length=2048, val_split=0.05):
    """Prepare the SYNTHETIC-1 dataset with validation split"""
    print("\nPreparing dataset...")
    
    # First explore the dataset structure
    dataset = explore_dataset_structure()
    
    # Split into train and validation
    print(val_split)
    dataset = dataset["train"].train_test_split(test_size=val_split, seed=42)
    print(f"Split into train ({len(dataset['train'])}) and validation ({len(dataset['test'])}) sets")
    
    # Format conversations
    print("\nFormatting conversations...")
    formatted_dataset = dataset.map( 
        format_conversation,
        num_proc=8,
        desc="Formatting conversations"
    )
    
    # Check if any examples have empty formatted text
    empty_count_train = sum(1 for example in formatted_dataset['train'] if len(example.get('formatted_text', '')) == 0)
    if empty_count_train > 0:
        print(f"WARNING: {empty_count_train} examples in train set have empty formatted_text!")
    
    # Tokenize the dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=formatted_dataset["train"].column_names,
        num_proc=8,
        desc="Tokenizing dataset"
    )
    
    
    # Additional validation after tokenization and filtering
    if len(tokenized_dataset['train']) == 0 or len(tokenized_dataset['test']) == 0:
        raise ValueError("Dataset is empty after tokenization! Consider adjusting min_length or check tokenization.")
    
    return tokenized_dataset

def format_conversation(example):
    """Convert the messages format to a simple text format"""
    try:
        if not isinstance(example, dict) or 'messages' not in example:
            print(f"Example missing 'messages' field: {example.keys()}")
            return {"formatted_text": ""}
            
        messages = example.get('messages', [])
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"Invalid messages format: {type(messages)}")
            return {"formatted_text": ""}
        
        formatted = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            
            # Add proper formatting to distinguish user/assistant roles
            if role == "user":
                formatted += f"USER: {content}\n\n"
            elif role == "assistant":
                # Check if it has thinking section to preserve it
                if "<think>" in content and "</think>" in content:
                    formatted += f"ASSISTANT: {content}\n\n"
                else:
                    formatted += f"ASSISTANT: {content}\n\n"
        
        # Add debugging to see if we're getting valid formatted text
        if len(formatted) < 10:
            print(f"WARNING: Short formatted text: '{formatted}'")
            
        return {"formatted_text": formatted}
    except Exception as e:
        print(f"Error formatting example: {e}")
        return {"formatted_text": ""}

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
    """Create a data collator with dynamic padding"""
    print("\nCreating data collator...")
    
    def collate_fn(examples):
        try:
            # Check if examples are valid
            if not examples or len(examples) == 0:
                raise ValueError("Empty batch received")
            
            # Pad sequences to the same length within this batch
            max_length_in_batch = max(len(example["input_ids"]) for example in examples)
            
            # Create padded tensors
            batch = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for example in examples:
                # Get current lengths
                curr_len = len(example["input_ids"])
                padding_len = max_length_in_batch - curr_len
                
                # Pad if necessary
                if padding_len > 0:
                    input_ids = example["input_ids"] + [tokenizer.pad_token_id] * padding_len
                    attention_mask = example["attention_mask"] + [0] * padding_len
                else:
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                
                # Add to batch
                batch["input_ids"].append(torch.tensor(input_ids))
                batch["attention_mask"].append(torch.tensor(attention_mask))
                batch["labels"].append(torch.tensor(input_ids))  # For causal LM, labels are the same as input_ids
            
            # Stack tensors
            batch["input_ids"] = torch.stack(batch["input_ids"])
            batch["attention_mask"] = torch.stack(batch["attention_mask"])
            batch["labels"] = torch.stack(batch["labels"])
            
            return batch
        except Exception as e:
            print(f"Error in collate function: {e}")
            raise
    
    # Test the collator with a small batch
    try:
        print("Testing data collator with sample batch...")
        sample_batch = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},  # Different length to test padding
        ]
        test_batch = collate_fn(sample_batch)
        print(f"Data collator test successful. Batch shapes: input_ids={test_batch['input_ids'].shape}, attention_mask={test_batch['attention_mask'].shape}")
    except Exception as e:
        print(f"Warning: Data collator test failed: {e}")
        print("This could indicate issues during training.")
    
    return collate_fn

# If you want to run this file directly for exploration
if __name__ == "__main__":
    print("Running data exploration...")
    explore_dataset_structure()