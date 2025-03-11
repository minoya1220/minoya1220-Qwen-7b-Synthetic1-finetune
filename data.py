import os
import json
import torch
from datasets import load_dataset
from tqdm import tqdm

def explore_dataset_structure():
    """Explore the SYNTHETIC-1 dataset structure properly"""
    print("\n===== EXPLORING DATASET STRUCTURE =====")
    
    dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
    print(f"Dataset splits: {list(dataset.keys())}")
    
    example = dataset['train'][0]
    print(f"\nTop-level keys: {list(example.keys())}")
    
    # Properly explore the messages field
    if 'messages' in example:
        print("\n----- Messages Structure -----")
        messages = example['messages']
        print(f"Number of messages: {len(messages)}")
        
        for i, msg in enumerate(messages):
            print(f"\nMessage {i}:")
            print(f"Role: {msg.get('role', 'unknown')}")
            content = msg.get('content', '')
            print(f"Content preview: {content[:100]}...")
            
            # Check if this is an assistant message with thinking
            if msg.get('role') == 'assistant' and '<think>' in content:
                print("Contains thinking section: Yes")
    
    return dataset

def download_dataset_locally(output_dir="./dataset_files"):
    """Download the SYNTHETIC-1 dataset to local machine"""
    print(f"\nDownloading dataset to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        dataset = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data")
        print(f"Dataset loaded successfully: {dataset is not None}")
        
        if 'train' in dataset:
            # Save dataset info
            with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
                info = {
                    "total_examples": len(dataset['train']),
                    "keys": list(dataset['train'][0].keys()),
                    "sample_message_count": len(dataset['train'][0]['messages'])
                }
                json.dump(info, f, indent=2)
            
            # Save a sample of examples (first 100)
            sample_size = min(100, len(dataset['train']))
            
            # Save each example as a separate JSON file
            for i in range(sample_size):
                example = dataset['train'][i]
                with open(os.path.join(output_dir, f"example_{i}.json"), "w") as f:
                    json.dump(example, f, indent=2)
            
            # Save a few complete examples in a single file
            with open(os.path.join(output_dir, "sample_examples.json"), "w") as f:
                samples_list = dataset['train'][:sample_size]
                json.dump(samples_list, f, indent=2)
            
            print(f"Dataset downloaded successfully to {output_dir}")
            print(f"Saved {sample_size} individual examples and dataset info")
            return dataset
        
        else:
            print("Error: 'train' split not found in dataset")
            return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def prepare_dataset(tokenizer, max_length=2048, val_split=0.05):
    """Prepare the SYNTHETIC-1 dataset with validation split"""
    print("\nPreparing dataset...")
    
    # First explore the dataset structure
    dataset = explore_dataset_structure()
    
    # Split into train and validation
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
    
    # Filter out examples that are too short
    min_length = 10
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: len(example.get('input_ids', [])) >= min_length
    )
    print(f"Final dataset: Train ({len(tokenized_dataset['train'])}), Validation ({len(tokenized_dataset['test'])})")
    
    # Additional validation after tokenization and filtering
    if len(tokenized_dataset['train']) == 0 or len(tokenized_dataset['test']) == 0:
        raise ValueError("Dataset is empty after tokenization and filtering! Consider adjusting min_length or check tokenization.")
    
    return tokenized_dataset

def format_conversation(example):
    """Convert the messages format to a simple text format"""
    try:
        if not isinstance(example, dict) or 'messages' not in example:
            return {"formatted_text": ""}
            
        messages = example.get('messages', [])
        if not isinstance(messages, list) or len(messages) == 0:
            return {"formatted_text": ""}
        
        formatted = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if not isinstance(role, str):
                role = str(role)
            if not isinstance(content, str):
                content = str(content)
            
            formatted += f"{role}: {content}\n\n"
        
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