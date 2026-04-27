"""
Model definitions for multimodal contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class GRAM3ModalLoss(nn.Module):
    """GRAM loss for 3-modal contrastive learning"""
    
    def __init__(self, contra_temp=0.07, lambda_dam=0.5):
        """
        Initialize GRAM loss
        
        Args:
            contra_temp: Temperature for contrastive loss
            lambda_dam: Weight for DAM loss component
        """
        super().__init__()
        self.contra_temp = contra_temp
        self.lambda_dam = lambda_dam

    def compute_volume(self, SMILES, modal1, modal2):
        """
        Compute volume using Gram determinant
        
        Args:
            SMILES: SMILES embeddings
            modal1: First modal embeddings
            modal2: Second modal embeddings
            
        Returns:
            torch.Tensor: Volume values
        """
        ll = torch.einsum('bd,bd->b', SMILES, SMILES)[:, None]
        l1 = SMILES @ modal1.T
        l2 = SMILES @ modal2.T
        
        m1m1 = torch.einsum('bd,bd->b', modal1, modal1)[None, :]
        m1m2 = torch.einsum('bd,bd->b', modal1, modal2)[None, :]
        m2m2 = torch.einsum('bd,bd->b', modal2, modal2)[None, :]
        
        G = torch.stack([
            torch.stack([ll.expand_as(l1), l1, l2], dim=-1),
            torch.stack([l1, m1m1.expand_as(l1), m1m2.expand_as(l1)], dim=-1),
            torch.stack([l2, m1m2.expand_as(l1), m2m2.expand_as(l1)], dim=-1)
        ], dim=-2)
        
        det = torch.det(G.float() + torch.eye(3, device=G.device)*1e-6)
        return torch.sqrt(torch.abs(det) + 1e-6)

    def forward(self, text_feats, smiles_feats, category_feats, rank, world_size):
        """
        Forward pass for GRAM loss
        
        Args:
            text_feats: Text feature embeddings
            smiles_feats: SMILES feature embeddings
            category_feats: Category feature embeddings
            rank: Current process rank
            world_size: Total number of processes
            
        Returns:
            torch.Tensor: Total loss
        """
        # Gather features from all processes
        text_all = self.concat_all_gather(text_feats)
        smiles_all = self.concat_all_gather(smiles_feats)
        category_all = self.concat_all_gather(category_feats)
        
        # Compute volume-based similarities
        volume = self.compute_volume(smiles_feats, text_all, category_all) / self.contra_temp
        volumeT = self.compute_volume(smiles_all, text_feats, category_feats).T / self.contra_temp
        
        B = text_feats.size(0)
        targets = torch.arange(B, device=text_feats.device) + rank * B
        
        # Volume-based contrastive loss
        loss_d2a = F.cross_entropy(-volume, targets, label_smoothing=0.1)
        loss_a2d = F.cross_entropy(-volumeT, targets, label_smoothing=0.1)
        contra_loss = (loss_d2a + loss_a2d) / 2

        # Standard similarity-based loss
        sim_t2s = text_feats @ smiles_all.T / self.contra_temp
        sim_s2t = smiles_feats @ text_all.T / self.contra_temp
        
        loss_t2s = F.cross_entropy(sim_t2s, targets, label_smoothing=0.1)
        loss_s2t = F.cross_entropy(sim_s2t, targets, label_smoothing=0.1)
        itm_loss = (loss_t2s + loss_s2t) / 2

        total_loss = contra_loss + self.lambda_dam * itm_loss
        return total_loss

    @staticmethod
    def concat_all_gather(tensor):
        """Gather tensors from all processes"""
        if dist.is_initialized():
            gathered = torch.distributed.nn.functional.all_gather(tensor)
            return torch.cat(gathered, dim=0)
        else:
            return tensor


class MultiModalContrastiveModel(nn.Module):
    """Multi-modal contrastive learning model with momentum-based integration"""
    
    def __init__(self, temperature=0.07, projection_dim=512, freeze_encoders=False, 
                 freeze_smiles_encoder=False, freeze_text_encoder=False, smile_tokenizer=None,
                 text_tokenizer=None, smiles_encoder=None, text_encoder=None, device=None, 
                 beta=0.9, initial_alpha=0.5):
        """
        Initialize multimodal contrastive model
        
        Args:
            temperature: Temperature for contrastive loss
            projection_dim: Dimension of projection layers
            freeze_encoders: Whether to freeze both encoders
            freeze_smiles_encoder: Whether to freeze SMILES encoder
            freeze_text_encoder: Whether to freeze text encoder
            smile_tokenizer: SMILES tokenizer
            text_tokenizer: Text tokenizer
            smiles_encoder: SMILES encoder model
            text_encoder: Text encoder model
            device: Device to run on
            beta: Momentum parameter
            initial_alpha: Initial momentum coefficient
        """
        super().__init__()
        self.temperature = temperature
        self.smile_tokenizer = smile_tokenizer
        self.text_tokenizer = text_tokenizer
        self.smiles_encoder = smiles_encoder
        self.text_encoder = text_encoder
        self.device = device
        
        # Momentum-based integration parameters
        self.beta = beta
        self.alpha = initial_alpha
        
        # Loss function
        self.gram_loss = GRAM3ModalLoss(
            contra_temp=temperature,
            lambda_dam=0.5  
        )
        
        # Freeze encoders if specified
        if freeze_smiles_encoder:
            self.smiles_encoder.eval()
            for param in self.smiles_encoder.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Projection layers
        self.smiles_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, projection_dim)
        )

    def update_alpha(self, global_loss, local_loss):
        """
        Update alpha using exponential moving average
        
        Args:
            global_loss: Current global loss value
            local_loss: Current local loss value
            
        Returns:
            float: Updated alpha value
        """
        if not isinstance(local_loss, torch.Tensor) or local_loss.item() == 0:
            return self.alpha
            
        # Calculate loss ratio
        loss_ratio = global_loss.item() / (global_loss.item() + local_loss.item())
        
        # Update with exponential moving average
        updated_alpha = self.beta * self.alpha + (1 - self.beta) * loss_ratio
        
        return updated_alpha

    def encode_smiles(self, inputs, return_hidden_states=False):
        """Encode SMILES sequences"""
        outputs = self.smiles_encoder(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0]
        projected = self.smiles_projector(pooled_output)
        
        if return_hidden_states:
            return nn.functional.normalize(projected, dim=1), outputs.last_hidden_state
        return nn.functional.normalize(projected, dim=1)

    def encode_text(self, inputs, return_hidden_states=False):
        """Encode text sequences"""
        outputs = self.text_encoder.encoder(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0]
        projected = self.text_projector(pooled_output)
        
        if return_hidden_states:
            return nn.functional.normalize(projected, dim=1), outputs.last_hidden_state
        return nn.functional.normalize(projected, dim=1)

    def encode_modalities(self, smiles, texts, category, return_hidden_states=False):
        """Encode all modalities in parallel"""
        outputs = {}
        
        if return_hidden_states:
            outputs['smiles'], smiles_hidden = self.encode_smiles(smiles, return_hidden_states=True)
            outputs['text'], text_hidden = self.encode_text(texts, return_hidden_states=True)
            outputs['category'] = self.encode_text(category)
            outputs['smiles_hidden'] = smiles_hidden
            outputs['text_hidden'] = text_hidden
        else:
            outputs['smiles'] = self.encode_smiles(smiles)
            outputs['text'] = self.encode_text(texts)
            outputs['category'] = self.encode_text(category)
            
        return outputs

    def compute_fg_local_loss(self, fg_data, rank=0, world_size=1):
        """
        Compute functional group local loss
        
        Args:
            fg_data: Functional group data
            rank: Current process rank
            world_size: Total number of processes
            
        Returns:
            torch.Tensor: Local loss value
        """
        if fg_data['fg_smiles_encoded'] is None or 'fg_counts_tensor' not in fg_data or fg_data['fg_counts_tensor'] is None:
            return torch.tensor(0.0, device=self.device)
        
        device = self.device

        fg_smiles = {k: v.to(device) for k, v in fg_data['fg_smiles_encoded'].items()}
        fg_texts = {k: v.to(device) for k, v in fg_data['fg_descriptions_encoded'].items()}
        fg_counts = fg_data['fg_counts_tensor'].to(device)
        molecule_to_fg_indices = fg_data['molecule_to_fg_indices']
        
        # Encode functional groups
        if hasattr(self, 'freeze_smiles_encoder') and self.freeze_smiles_encoder:
            with torch.no_grad():
                fg_smiles_outputs = self.smiles_encoder(**fg_smiles)
                fg_smiles_hidden = fg_smiles_outputs.last_hidden_state[:, 0]
        else:
            fg_smiles_outputs = self.smiles_encoder(**fg_smiles)
            fg_smiles_hidden = fg_smiles_outputs.last_hidden_state[:, 0]
            
        if hasattr(self, 'freeze_text_encoder') and self.freeze_text_encoder:
            with torch.no_grad():
                fg_texts_outputs = self.text_encoder.encoder(**fg_texts)
                fg_texts_hidden = fg_texts_outputs.last_hidden_state[:, 0]
        else:
            fg_texts_outputs = self.text_encoder.encoder(**fg_texts)
            fg_texts_hidden = fg_texts_outputs.last_hidden_state[:, 0]
        
        # Project and normalize
        fg_smiles_embeds = self.smiles_projector(fg_smiles_hidden)
        fg_texts_embeds = self.text_projector(fg_texts_hidden)
        
        fg_smiles_embeds = nn.functional.normalize(fg_smiles_embeds, dim=1)
        fg_texts_embeds = nn.functional.normalize(fg_texts_embeds, dim=1)
        
        # Pool functional groups by molecule
        batch_size = len(molecule_to_fg_indices)
        proj_dim = fg_smiles_embeds.shape[1]
        
        pooled_fg_smiles = torch.zeros(batch_size, proj_dim, device=device)
        pooled_fg_texts = torch.zeros(batch_size, proj_dim, device=device)
        valid_mol_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for mol_idx, fg_indices in enumerate(molecule_to_fg_indices):
            if not fg_indices:
                continue
            
            mol_fg_smiles_embeds = fg_smiles_embeds[fg_indices]
            mol_fg_texts_embeds = fg_texts_embeds[fg_indices]
            mol_fg_counts = fg_counts[fg_indices].unsqueeze(1)
            
            # Weight by count
            weighted_smiles = mol_fg_smiles_embeds * mol_fg_counts
            weighted_texts = mol_fg_texts_embeds * mol_fg_counts
            
            if mol_fg_counts.sum() > 0:
                pooled_fg_smiles[mol_idx] = weighted_smiles.sum(dim=0) / mol_fg_counts.sum()
                pooled_fg_texts[mol_idx] = weighted_texts.sum(dim=0) / mol_fg_counts.sum()
                valid_mol_mask[mol_idx] = True
        
        # Filter valid molecules
        valid_indices = valid_mol_mask.nonzero().squeeze(-1)
        if valid_indices.numel() == 0:
            return torch.tensor(0.0, device=device)
        
        pooled_fg_smiles = pooled_fg_smiles[valid_indices]
        pooled_fg_texts = pooled_fg_texts[valid_indices]
        
        pooled_fg_smiles = nn.functional.normalize(pooled_fg_smiles, dim=1)
        pooled_fg_texts = nn.functional.normalize(pooled_fg_texts, dim=1)
        
        # Use local data for loss computation
        pooled_fg_smiles_all = pooled_fg_smiles
        pooled_fg_texts_all = pooled_fg_texts
        valid_count = pooled_fg_smiles.size(0)
        
        # Compute similarities and loss
        sim_fg_s2t = torch.matmul(pooled_fg_smiles, pooled_fg_texts_all.t()) / self.temperature
        sim_fg_t2s = torch.matmul(pooled_fg_texts, pooled_fg_smiles_all.t()) / self.temperature
        
        targets = torch.arange(valid_count, device=device)

        if targets.max() >= sim_fg_s2t.size(1): 
            targets = torch.clamp(targets, 0, sim_fg_s2t.size(1) - 1)
        
        loss_fg_s2t = F.cross_entropy(sim_fg_s2t, targets, label_smoothing=0.1)
        loss_fg_t2s = F.cross_entropy(sim_fg_t2s, targets, label_smoothing=0.1)
        
        fg_local_loss = (loss_fg_s2t + loss_fg_t2s) / 2
        
        return fg_local_loss

    def forward(self, smiles, texts, category, fg_data, current_epoch=0, total_epochs=10):
        """
        Forward pass
        
        Args:
            smiles: SMILES input
            texts: Text input
            category: Category input
            fg_data: Functional group data
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            
        Returns:
            tuple: (total_loss, global_loss, local_loss, alpha)
        """
        # Encode modalities
        embeddings = self.encode_modalities(smiles, texts, category, return_hidden_states=True)
        
        # Compute global loss
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        global_loss = self.gram_loss(
            text_feats=embeddings['text'],
            smiles_feats=embeddings['smiles'],
            category_feats=embeddings['category'],
            rank=rank,
            world_size=world_size
        )
        
        # Compute local loss
        fg_local_loss = self.compute_fg_local_loss(fg_data, rank, world_size)
        
        # Update alpha
        self.alpha = self.update_alpha(global_loss, fg_local_loss)
        
        # Combine losses
        if isinstance(fg_local_loss, torch.Tensor) and fg_local_loss.item() > 0:
            total_loss = self.alpha * global_loss + (1 - self.alpha) * fg_local_loss
        else:
            total_loss = global_loss
        
        return total_loss, global_loss, fg_local_loss, self.alpha