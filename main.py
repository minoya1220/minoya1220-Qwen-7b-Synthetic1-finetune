import argparse
import sys
import torch

from utils import print_device_info
from train import train

if __name__ == "__main__":
    print("\n=== Qwen-7B Synthetic1 SFT Training ===")
    print_device_info()
    
    parser = argparse.ArgumentParser(description="Train Qwen model on SYNTHETIC-1 dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-7B", help="Model name")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=float, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
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