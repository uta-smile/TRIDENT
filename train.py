"""
Training functions and main worker for multimodal contrastive learning
"""

import os
import sys
import time
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration

from config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_PATHS, DATA_PATHS
from dataset import MoleculeTextDataset, CollateFunction
from torch.utils.data import Subset
from models import MultiModalContrastiveModel
from utils import (
    setup_ddp, cleanup_ddp, setup_signal_handlers, save_checkpoint,
    get_device, print_model_info, format_time, AverageMeter, create_save_directory
)


def process_batch_data(batch, device):
    """
    Process and move batch data to device
    
    Args:
        batch: Batch data from dataloader
        device: Target device
        
    Returns:
        tuple: Processed batch data
    """
    smiles_encoded = batch['smiles_encoded']
    text_encoded = batch['text_encoded']
    category_encoded = batch['category_encoded']
    fg_data = batch['fg_data']
    
    # Move encoded data to device
    smiles_encoded = {k: v.to(device) for k, v in smiles_encoded.items()}
    text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
    category_encoded = {k: v.to(device) for k, v in category_encoded.items()}
    
    # Process functional group data
    processed_fg_data = {
        'molecule_to_fg_indices': fg_data['molecule_to_fg_indices'],
    }
    
    if fg_data['fg_smiles_encoded'] is not None:
        processed_fg_data['fg_smiles_encoded'] = {k: v.to(device) for k, v in fg_data['fg_smiles_encoded'].items()}
    else:
        processed_fg_data['fg_smiles_encoded'] = None
    
    if fg_data['fg_descriptions_encoded'] is not None:
        processed_fg_data['fg_descriptions_encoded'] = {k: v.to(device) for k, v in fg_data['fg_descriptions_encoded'].items()}
    else:
        processed_fg_data['fg_descriptions_encoded'] = None
    
    if fg_data['fg_counts_tensor'] is not None:
        processed_fg_data['fg_counts_tensor'] = fg_data['fg_counts_tensor'].to(device)
    else:
        processed_fg_data['fg_counts_tensor'] = None

    return smiles_encoded, text_encoded, category_encoded, processed_fg_data


def train_epoch(model, train_loader, optimizer, device, epoch, rank, total_epochs):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        rank: Process rank
        total_epochs: Total number of epochs
        
    Returns:
        dict: Training metrics
    """
    model.train()
    
    # Initialize meters
    loss_meter = AverageMeter('Loss', ':.6f')
    global_loss_meter = AverageMeter('Global', ':.6f')
    local_loss_meter = AverageMeter('Local', ':.6f')
    alpha_meter = AverageMeter('Alpha', ':.4f')
    
    start_time = time.time()
    
    try:
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
        else:
            pbar = train_loader
            
        for batch_idx, batch in enumerate(pbar):
            # Process batch data
            smiles_encoded, text_encoded, category_encoded, processed_fg_data = process_batch_data(batch, device)

            # Forward pass
            loss, global_loss, local_loss, alpha = model(
                smiles_encoded, 
                text_encoded, 
                category_encoded, 
                processed_fg_data,
                current_epoch=epoch,
                total_epochs=total_epochs
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update meters
            batch_size = smiles_encoded['input_ids'].size(0)
            loss_meter.update(loss.item(), batch_size)
            global_loss_meter.update(global_loss.item(), batch_size)
            
            if isinstance(local_loss, torch.Tensor):
                local_loss_value = local_loss.item()
            else:
                local_loss_value = local_loss
            local_loss_meter.update(local_loss_value, batch_size)
            alpha_meter.update(alpha, batch_size)
            
            # Update progress bar
            if rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Global': f'{global_loss.item():.4f}',
                    'Local': f'{local_loss_value:.4f}',
                    'Alpha': f'{alpha:.4f}'
                })
                
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
        sys.exit(1)
    
    epoch_time = time.time() - start_time
    
    # Gather metrics across all processes
    if dist.is_initialized():
        dist.barrier()
    
    metrics = {
        'loss': loss_meter.avg,
        'global_loss': global_loss_meter.avg,
        'local_loss': local_loss_meter.avg,
        'alpha': alpha_meter.avg,
        'epoch_time': epoch_time
    }
    
    if rank == 0:
        print(f'Epoch {epoch+1}/{total_epochs} completed in {format_time(epoch_time)}')
        print(f'Average Loss: {metrics["loss"]:.6f}, '
              f'Global: {metrics["global_loss"]:.6f}, '
              f'Local: {metrics["local_loss"]:.6f}, '
              f'Alpha: {metrics["alpha"]:.4f}')
    
    return metrics


def create_model_and_tokenizers():
    """
    Create model and tokenizers
    
    Returns:
        tuple: (smile_tokenizer, smile_model, text_tokenizer, text_model)
    """
    # Load SMILES tokenizer and model
    smile_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS['smiles_model'], 
        trust_remote_code=True
    )
    smile_model = AutoModel.from_pretrained(
        MODEL_PATHS['smiles_model'], 
        trust_remote_code=True
    )

    # Load text tokenizer and model
    text_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS['text_model'], 
        model_max_length=TRAIN_CONFIG['max_length']
    )
    text_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATHS['text_model'])

    return smile_tokenizer, smile_model, text_tokenizer, text_model


def create_data_loader(dataset, tokenizers, batch_size, world_size, rank):
    """
    Create data loader with distributed sampler
    
    Args:
        dataset: Dataset object
        tokenizers: Tuple of (smile_tokenizer, text_tokenizer)
        batch_size: Batch size
        world_size: Number of processes
        rank: Current process rank
        
    Returns:
        DataLoader: Configured data loader
    """
    smile_tokenizer, text_tokenizer = tokenizers
    
    # Create distributed sampler
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # Create collate function
    collate_fn = CollateFunction(
        tokenizer_smiles=smile_tokenizer,
        tokenizer_text=text_tokenizer,
        max_length=TRAIN_CONFIG['max_length'],
        fg_smiles_path=DATA_PATHS['fg_smiles'],
        fg_descriptions_path=DATA_PATHS['fg_descriptions']
    )

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=TRAIN_CONFIG['num_workers'],
        prefetch_factor=TRAIN_CONFIG['prefetch_factor'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        multiprocessing_context="spawn"
    )
    
    return train_loader, train_sampler


def main_worker(rank, world_size, batch_size, num_epochs, beta=0.9, initial_alpha=0.5):
    """
    Main worker function for distributed training
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        batch_size: Batch size per GPU
        num_epochs: Number of training epochs
        beta: Momentum parameter
        initial_alpha: Initial momentum coefficient
    """
    # Setup
    setup_signal_handlers()
    setup_ddp(rank, world_size)
    device = get_device(rank)
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Configuration: batch_size={batch_size}, epochs={num_epochs}, beta={beta}, alpha={initial_alpha}")
    
    try:
        # Create tokenizers and models
        smile_tokenizer, smile_model, text_tokenizer, text_model = create_model_and_tokenizers()
        
        # Load dataset
        dataset = MoleculeTextDataset(data_path=DATA_PATHS['train_data'])

        # # For test
        # dataset = Subset(dataset, range(100))
        
        if rank == 0:
            print(f"Loaded dataset with {len(dataset)} samples")
        
        # Create data loader
        train_loader, train_sampler = create_data_loader(
            dataset, (smile_tokenizer, text_tokenizer), batch_size, world_size, rank
        )
        
        # Create model
        model = MultiModalContrastiveModel(
            temperature=MODEL_CONFIG['temperature'],
            projection_dim=MODEL_CONFIG['projection_dim'],
            freeze_smiles_encoder=True,
            freeze_text_encoder=True,
            smile_tokenizer=smile_tokenizer,
            text_tokenizer=text_tokenizer,
            smiles_encoder=smile_model,
            text_encoder=text_model,
            device=device,
            beta=beta,
            initial_alpha=initial_alpha
        ).to(device)

        # Print model information
        print_model_info(model, rank)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank])
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
        
        # Create save directory
        if rank == 0:
            create_save_directory(DATA_PATHS['save_dir'])
        
        # Training loop
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            
            # Train for one epoch
            metrics = train_epoch(model, train_loader, optimizer, device, epoch, rank, num_epochs)
            
            # Save checkpoint
            if rank == 0 and (epoch + 1) % 2 == 0:
                save_path = os.path.join(
                    DATA_PATHS['save_dir'],
                    f'beta{beta}_a{initial_alpha}_epoch_{epoch+1}.pt'
                )
                additional_info = {
                    'beta': beta,
                    'alpha': model.module.alpha,
                    'metrics': metrics
                }
                save_checkpoint(model, optimizer, epoch, metrics['loss'], save_path, additional_info)
            
            # Synchronize processes
            if dist.is_initialized():
                dist.barrier()

        if rank == 0:
            print("Training completed successfully!")
            
    except Exception as e:
        print(f"Error in main_worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()


def train_model(batch_size=None, epochs=None, beta=None, initial_alpha=None):
    """
    Public interface for training the model
    
    Args:
        batch_size: Batch size per GPU (uses config default if None)
        epochs: Number of epochs (uses config default if None)
        beta: Momentum parameter (uses config default if None)
        initial_alpha: Initial alpha (uses config default if None)
    """
    import torch.multiprocessing as mp
    
    # Use config defaults if not specified
    batch_size = batch_size or TRAIN_CONFIG['batch_size']
    epochs = epochs or TRAIN_CONFIG['epochs']
    beta = beta or MODEL_CONFIG['beta']
    initial_alpha = initial_alpha or MODEL_CONFIG['initial_alpha']
    
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No CUDA devices available for training")
    
    print(f"Starting distributed training with {world_size} GPUs")
    
    mp.spawn(
        main_worker,
        args=(world_size, batch_size, epochs, beta, initial_alpha),
        nprocs=world_size
    )