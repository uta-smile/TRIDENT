"""
Dataset classes for molecular and text data processing
"""

import json
import inspect
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit import RDLogger

from config import RDKIT_CONFIG

# Disable RDKit logging if configured
if RDKIT_CONFIG.get('disable_logs', True):
    RDLogger.DisableLog('rdApp.*')


class MoleculeTextDataset(Dataset):
    """Dataset for molecule-text pairs with functional group analysis"""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSON file containing molecule data
        """
        with open(data_path, 'r') as file:
            self.data = json.load(file)
        
        self.data_list = [{"CID": cid, **info} for cid, info in self.data.items()]
        
        # Store functional group function names (not the functions themselves)
        self.fragment_function_names = []
        for name, func in inspect.getmembers(Fragments, inspect.isfunction):
            if name.startswith('fr_'):
                self.fragment_function_names.append(name)
    
    def __len__(self):
        return len(self.data_list)

    def get_functional_groups(self, smiles):
        """
        Extract functional groups from SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict: Functional group counts
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {}
        
        results = {}
        # Get functions dynamically when needed instead of storing them
        for name in self.fragment_function_names:
            try:
                func = getattr(Fragments, name)
                count = func(mol)
                if count > 0:
                    results[name] = count
            except:
                continue
        return results

    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Args:
            idx: Index
            
        Returns:
            tuple: (smiles, description, hta, fragment_counts)
        """
        item = self.data_list[idx]
        smiles = item.get('SMILES', "No SMILES available")
        description = item.get('Description', "No description available")
        hta = item.get('HTA', "No HTA information available")
        
        fragment_counts = self.get_functional_groups(smiles) if smiles != "No SMILES available" else {}

        return smiles, description, hta, fragment_counts


class CollateFunction:
    """Collate function for batching molecular data"""
    
    def __init__(self, tokenizer_smiles, tokenizer_text, max_length, 
                 fg_smiles_path='pretrain_data/fg/fg_SMILES.csv', 
                 fg_descriptions_path='pretrain_data/fg/fg_text.json'):
        """
        Initialize collate function
        
        Args:
            tokenizer_smiles: SMILES tokenizer
            tokenizer_text: Text tokenizer
            max_length: Maximum sequence length
            fg_smiles_path: Path to functional group SMILES mapping
            fg_descriptions_path: Path to functional group descriptions
        """
        self.tokenizer_smiles = tokenizer_smiles
        self.tokenizer_text = tokenizer_text
        self.max_length = max_length
        
        # Load functional group SMILES mapping
        self.fg_smiles_map = {}
        try:
            fg_smiles_df = pd.read_csv(fg_smiles_path)
            for _, row in fg_smiles_df.iterrows():
                self.fg_smiles_map[row['Functional Group']] = row['SMILES']
        except Exception as e:
            print(f"Warning: Could not load functional group SMILES mapping: {e}")
        
        # Load functional group descriptions
        self.fg_descriptions_map = {}
        try:
            with open(fg_descriptions_path, 'r') as f:
                self.fg_descriptions_map = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load functional group descriptions: {e}")
    
    def encode_functional_groups(self, fragment_counts_list):
        """
        Encode functional groups for the batch
        
        Args:
            fragment_counts_list: List of functional group counts for each molecule
            
        Returns:
            dict: Encoded functional group data
        """
        batch_size = len(fragment_counts_list)
        
        all_fg_smiles = []
        all_fg_descriptions = []
        all_fg_counts = []
        molecule_to_fg_indices = []
        
        current_idx = 0
        for mol_idx, fragment_counts in enumerate(fragment_counts_list):
            mol_fg_indices = []
            
            for fg_name, count in fragment_counts.items():
                # Get functional group SMILES
                fg_smiles = self.fg_smiles_map.get(fg_name, "")
                if not fg_smiles and fg_name.startswith("fr_"):
                    alt_name = fg_name[3:]
                    fg_smiles = self.fg_smiles_map.get(alt_name, "")
                
                # Get functional group description
                fg_description = self.fg_descriptions_map.get(fg_name, "")
                if not fg_description and fg_name.startswith("fr_"):
                    alt_name = fg_name[3:]
                    fg_description = self.fg_descriptions_map.get(alt_name, "")
                
                # Generate default description if none found
                if not fg_description:
                    readable_name = fg_name
                    if fg_name.startswith("fr_"):
                        readable_name = fg_name[3:].replace("_", " ")
                    fg_description = f"The functional group {readable_name}"
                
                all_fg_smiles.append(fg_smiles)
                all_fg_descriptions.append(fg_description)
                all_fg_counts.append(count)
                mol_fg_indices.append(current_idx)
                current_idx += 1
            
            molecule_to_fg_indices.append(mol_fg_indices)
        
        # Encode functional group SMILES
        fg_smiles_encoded = None
        if all_fg_smiles:
            fg_smiles_encoded = self.tokenizer_smiles(
                all_fg_smiles,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        # Encode functional group descriptions
        fg_descriptions_encoded = None
        if all_fg_descriptions:
            fg_descriptions_encoded = self.tokenizer_text(
                all_fg_descriptions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
        
        fg_counts_tensor = torch.tensor(all_fg_counts, dtype=torch.float) if all_fg_counts else None
        
        return {
            'fg_smiles_encoded': fg_smiles_encoded,
            'fg_descriptions_encoded': fg_descriptions_encoded,
            'fg_counts_tensor': fg_counts_tensor,
            'molecule_to_fg_indices': molecule_to_fg_indices
        }
    
    def __call__(self, batch):
        """
        Collate batch data
        
        Args:
            batch: Batch of samples
            
        Returns:
            dict: Collated batch data
        """
        smiles_list, text_list, category_list, fragment_counts_list = zip(*batch)
        
        # Encode SMILES
        smiles_encoded = self.tokenizer_smiles(
            list(smiles_list),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode text
        text_encoded = self.tokenizer_text(
            list(text_list),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Encode category
        category_encoded = self.tokenizer_text(
            list(category_list),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode functional groups
        fg_data = self.encode_functional_groups(fragment_counts_list)
        
        return {
            'smiles_encoded': smiles_encoded,
            'text_encoded': text_encoded,
            'category_encoded': category_encoded,
            'fragment_counts': fragment_counts_list,
            'fg_data': fg_data
        }