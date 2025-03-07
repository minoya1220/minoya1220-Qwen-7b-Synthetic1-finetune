import torch
import os
import wandb
from transformers import TrainerCallback

class WandbCallback(TrainerCallback):
    """Callback for logging metrics to Weights and Biases"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)

def get_device_info():
    """Get and print device information"""
    device_info = {
        "PyTorch version": torch.__version__,
        "CUDA available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "CUDA device": torch.cuda.get_device_name(0),
            "CUDA version": torch.version.cuda,
            "Number of GPUs": torch.cuda.device_count()
        })
    
    return device_info
    
def print_device_info():
    """Print device information"""
    device_info = get_device_info()
    
    print("=== Training Setup Information ===")
    for key, value in device_info.items():
        print(f"{key}: {value}")

def init_wandb(model_name, batch_size, learning_rate, num_epochs, max_length, gradient_accumulation_steps):
    """Initialize Weights and Biases logging"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        return True
    except Exception as e:
        print(f"Warning: WandB initialization failed: {e}")
        print("Training will continue without WandB logging.")
        return False

def create_output_dir(output_dir=None):
    """Create a timestamped output directory"""
    from datetime import datetime
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./synthetic1_output_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    return output_dir