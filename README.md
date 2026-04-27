# Multimodal Contrastive Learning

Molecular representation learning with SMILES, text, HTA, and functional groups.

## File Structure

```
├── main.py          # Main entry
├── config.py        # Configuration
├── dataset.py       # Data processing
├── models.py        # Model definitions
├── train.py         # Training logic
├── utils.py         # Utilities
├── data/            # Data directory
└── save_model/      # Model outputs
```

## Environment Setup

```bash
# Create a new conda environment
conda create -n TRIDENT python=3.10

# Activate the environment
conda activate TRIDENT

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic training
CUDA_VISIBLE_DEVICES=0,1 python main.py

# Custom settings
CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 32 --epochs 50
```