"""
Configuration file for multimodal contrastive learning
"""

# Model configuration
MODEL_CONFIG = {
    'temperature': 0.07,
    'projection_dim': 512,
    'beta': 0.9,  # Momentum parameter
    'initial_alpha': 0.5,  # Initial momentum coefficient
    'lambda_dam': 0.5,  # GRAM loss parameter
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 40,
    'epochs': 40,
    'learning_rate': 1e-5,
    'max_length': 512,
    'num_workers': 2,
    'prefetch_factor': 2,
    'label_smoothing': 0.1,
}

# Model paths
MODEL_PATHS = {
    'smiles_model': "ibm/MoLFormer-XL-both-10pct",
    'text_model': "laituan245/molt5-base",
    # Alternative text model: "allenai/scibert_scivocab_uncased"
}

# Data paths
DATA_PATHS = {
    'train_data': 'pretrain_data/SMILES_Text_HTA/STH.json',
    'fg_smiles': 'pretrain_data/fg/fg_SMILES.csv',
    'fg_descriptions': 'pretrain_data/fg/fg_text.json',
    'save_dir': 'save_model/pretrained_model',
}

# DDP configuration
DDP_CONFIG = {
    'master_addr': 'localhost',
    'master_port': '12355',
    'backend': 'nccl',
}

# RDKit configuration
RDKIT_CONFIG = {
    'disable_logs': True,  # Disable RDKit logging
}