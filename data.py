import torch
from datasets import load_dataset
from tqdm import tqdm
import os
import json

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
                # Save directly without converting to dict
                with open(os.path.join(output_dir, f"example_{i}.json"), "w") as f:
                    json.dump(example, f, indent=2)
            
            # Save a few complete examples in a single file
            with open(os.path.join(output_dir, "sample_examples.json"), "w") as f:
                # Save directly as a list without converting to dict
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
        try:
            # Format into chat format 
            messages = example['messages']
            
            # Verify messages structure
            if not isinstance(messages, list) or len(messages) == 0:
                print(f"Warning: Invalid messages format in example {example.get('response_id', 'unknown')}")
                return {"formatted_text": ""}
            
            formatted = ""
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    print(f"Warning: Invalid message format in example {example.get('response_id', 'unknown')}")
                    continue
                
                role = msg['role']
                content = msg['content']
                
                # Keep the thinking process intact - it's crucial for training
                formatted += f"{role}: {content}\n\n"
            
            return {"formatted_text": formatted}
        except Exception as e:
            print(f"Error formatting example {example.get('response_id', 'unknown')}: {e}")
            return {"formatted_text": ""}
    
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
        
        # Filter out empty examples
        print("\nFiltering out empty examples...")
        formatted_dataset = formatted_dataset.filter(
            lambda example: len(example['formatted_text']) > 0,
            num_proc=16,
            desc="Filtering empty examples"
        )
        print(f"After filtering - Train: {len(formatted_dataset['train'])}, Validation: {len(formatted_dataset['test'])}")
    except Exception as e:
        print(f"Error formatting dataset: {e}")
        raise
    
    def tokenize_function(examples):
        try:
            return tokenizer(
                examples["formatted_text"],
                max_length=max_length,
                truncation=True,
                padding=False  # We'll handle padding in the data collator
            )
        except Exception as e:
            print(f"Error tokenizing examples: {e}")
            # Return empty tokenized examples as fallback
            return {"input_ids": [[]], "attention_mask": [[]]} * len(examples["formatted_text"])
    
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
            
            # Filter out examples that are too short
            print("\nFiltering out examples that are too short...")
            min_length = 10  # Adjust as needed
            tokenized_dataset = tokenized_dataset.filter(
                lambda example: len(example['input_ids']) >= min_length,
                num_proc=16,
                desc="Filtering short examples"
            )
            print(f"After filtering - Train: {len(tokenized_dataset['train'])}, Validation: {len(tokenized_dataset['test'])}")
        else:
            print("Warning: No examples in tokenized dataset!")
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        raise
    
    print(f"Final train dataset size: {len(tokenized_dataset['train'])}")
    print(f"Final validation dataset size: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset

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