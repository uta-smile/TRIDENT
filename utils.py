"""
Utility functions for distributed training and model management
"""

import os
import sys
import signal
import torch
import torch.distributed as dist
from config import DDP_CONFIG


def setup_ddp(rank, world_size):
    """
    Initialize DDP process group
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = DDP_CONFIG['master_addr']
    os.environ['MASTER_PORT'] = DDP_CONFIG['master_port']
    dist.init_process_group(DDP_CONFIG['backend'], rank=rank, world_size=world_size, device_id=rank)


def cleanup_ddp():
    """Clean up DDP process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def signal_handler(signum, frame):
    """
    Handle interrupt signals gracefully
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    print(f"Received interrupt signal {signum}, cleaning up...")
    try:
        # Add timeout to avoid hanging
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        print("Cleanup completed, exiting...")
        os._exit(0)  # Force exit without cleanup


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def save_checkpoint(model, optimizer, epoch, loss, save_path, additional_info=None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        additional_info: Additional information to save (dict)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load to (optional)
        
    Returns:
        dict: Checkpoint information
    """
    if device:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def get_device(rank=None):
    """
    Get appropriate device for training
    
    Args:
        rank: Process rank (for multi-GPU training)
        
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        if rank is not None:
            return torch.device(f'cuda:{rank}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model, rank=0):
    """
    Print model information
    
    Args:
        model: PyTorch model
        rank: Process rank (only rank 0 prints)
    """
    if rank == 0:
        total_params, trainable_params = count_parameters(model)
        print(f"Model Information:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")


def format_time(seconds):
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def create_save_directory(save_dir):
    """
    Create save directory if it doesn't exist
    
    Args:
        save_dir: Directory path to create
    """
    os.makedirs(save_dir, exist_ok=True)