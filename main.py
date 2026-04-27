#!/usr/bin/env python3
"""
Main entry point for multimodal contrastive learning training
"""

import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, TRAIN_CONFIG
from train import train_model


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train multimodal contrastive model with momentum-based integration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=TRAIN_CONFIG['batch_size'],
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=TRAIN_CONFIG['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=TRAIN_CONFIG['learning_rate'],
        help='Learning rate'
    )
    
    # Model parameters
    parser.add_argument(
        '--beta', 
        type=float, 
        default=MODEL_CONFIG['beta'],
        help='Momentum parameter for alpha update'
    )
    parser.add_argument(
        '--initial_alpha', 
        type=float, 
        default=MODEL_CONFIG['initial_alpha'],
        help='Initial momentum coefficient for loss combination'
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=MODEL_CONFIG['temperature'],
        help='Temperature for contrastive loss'
    )
    parser.add_argument(
        '--projection_dim', 
        type=int, 
        default=MODEL_CONFIG['projection_dim'],
        help='Dimension of projection layers'
    )
    
    # Data parameters
    parser.add_argument(
        '--data_path', 
        type=str, 
        help='Path to training data (overrides config)'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        help='Directory to save models (overrides config)'
    )
    
    # Distributed training
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=TRAIN_CONFIG['num_workers'],
        help='Number of data loader workers'
    )
    
    # Debugging and logging
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """
    Validate command line arguments
    
    Args:
        args: Parsed arguments
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    if args.epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    if not (0 <= args.beta <= 1):
        raise ValueError("Beta must be between 0 and 1")
    
    if not (0 <= args.initial_alpha <= 1):
        raise ValueError("Initial alpha must be between 0 and 1")
    
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    if args.projection_dim <= 0:
        raise ValueError("Projection dimension must be positive")
    
    return True


def update_config(args):
    """
    Update global configuration with command line arguments
    
    Args:
        args: Parsed arguments
    """
    if args.data_path:
        from config import DATA_PATHS
        DATA_PATHS['train_data'] = args.data_path
    
    if args.save_dir:
        from config import DATA_PATHS
        DATA_PATHS['save_dir'] = args.save_dir
    
    # Update training config
    TRAIN_CONFIG['learning_rate'] = args.learning_rate
    TRAIN_CONFIG['num_workers'] = args.num_workers
    
    # Update model config
    MODEL_CONFIG['beta'] = args.beta
    MODEL_CONFIG['initial_alpha'] = args.initial_alpha
    MODEL_CONFIG['temperature'] = args.temperature
    MODEL_CONFIG['projection_dim'] = args.projection_dim


def print_configuration(args):
    """Print training configuration"""
    print("=" * 60)
    print("MULTIMODAL CONTRASTIVE LEARNING TRAINING")
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of workers: {args.num_workers}")
    print()
    print("Model Configuration:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Projection dimension: {args.projection_dim}")
    print(f"  Beta (momentum): {args.beta}")
    print(f"  Initial alpha: {args.initial_alpha}")
    print()
    
    if args.debug:
        print("Debug mode: ENABLED")
    if args.verbose:
        print("Verbose mode: ENABLED")
    print("=" * 60)


def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)
        
        # Update configuration
        update_config(args)
        
        # Print configuration
        print_configuration(args)
        
        # Check for CUDA availability
        import torch
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Training will be very slow on CPU.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
        
        world_size = torch.cuda.device_count()
        if world_size == 0:
            print("No CUDA devices found. Please ensure GPU drivers are installed.")
            return
        
        print(f"Found {world_size} CUDA device(s)")
        
        # Start training
        print("Starting training...")
        train_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            beta=args.beta,
            initial_alpha=args.initial_alpha
        )
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()