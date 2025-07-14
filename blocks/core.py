import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from transformers import AutoTokenizer, AutoModel
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from blocks.embed import CodeEmbedder

class Vec2VecResidualBlock(nn.Module): 
    """Residual block for Vec2VecAdapter internal process"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = (input_dim == output_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.norm( self.act(  self.layer(x) ) )

        if self.residual:
            out = out + x

        return out 
    

class Vec2VecAdapter(nn.Module):
    """Adapter module for vec2vec - transforms embeddings to/from universal space"""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            Vec2VecResidualBlock(input_dim, latent_dim*2),
            Vec2VecResidualBlock(latent_dim*2, latent_dim*2),
            Vec2VecResidualBlock(latent_dim*2, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            )
        
    def forward(self, x):
        return self.layers(x)

# update for latent space size, depth, and add residual connections
class Vec2VecBackbone(nn.Module):
    """Shared backbone network for universal latent space"""
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
            Vec2VecResidualBlock(latent_dim, latent_dim),
        )
        
    def forward(self, x):
        return self.layers(x)  

class Vec2VecDiscriminator(nn.Module):
    """Discriminator for adversarial training"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class CodeVec2Vec(nn.Module):
    """Main Vec2Vec model for code translation"""
    
    def __init__(self, embedding_dim: int, latent_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        
        # Input adapters (Python -> Universal, C -> Universal)
        self.adapter_py = Vec2VecAdapter(embedding_dim, latent_dim)
        self.adapter_c = Vec2VecAdapter(embedding_dim, latent_dim)
        
        # Shared backbone
        self.backbone = Vec2VecBackbone(latent_dim)
        
        # Output adapters (Universal -> Python, Universal -> C)
        self.output_py = Vec2VecAdapter(latent_dim, embedding_dim)
        self.output_c = Vec2VecAdapter(latent_dim, embedding_dim)
        
        # Discriminators
        self.disc_py = Vec2VecDiscriminator(embedding_dim)
        self.disc_c = Vec2VecDiscriminator(embedding_dim)
        self.disc_latent_py = Vec2VecDiscriminator(latent_dim)
        self.disc_latent_c = Vec2VecDiscriminator(latent_dim)
        
    def encode_python(self, py_embeddings):
        """Python -> Universal latent space"""
        adapted = self.adapter_py(py_embeddings)
        return self.backbone(adapted)
    
    def encode_c(self, c_embeddings):
        """C -> Universal latent space"""
        adapted = self.adapter_c(c_embeddings)
        return self.backbone(adapted)
    
    def decode_python(self, latent):
        """Universal latent space -> Python"""
        return self.output_py(latent)
    
    def decode_c(self, latent):
        """Universal latent space -> C"""
        return self.output_c(latent)
    
    def translate_py_to_c(self, py_embeddings):
        """Translate Python embeddings to C space"""
        latent = self.encode_python(py_embeddings)
        return self.decode_c(latent)
    
    def translate_c_to_py(self, c_embeddings):
        """Translate C embeddings to Python space"""
        latent = self.encode_c(c_embeddings)
        return self.decode_python(latent)
    
    def reconstruct_python(self, py_embeddings):
        """Reconstruct Python embeddings (for reconstruction loss)"""
        latent = self.encode_python(py_embeddings)
        return self.decode_python(latent)
    
    def reconstruct_c(self, c_embeddings):
        """Reconstruct C embeddings (for reconstruction loss)"""
        latent = self.encode_c(c_embeddings)
        return self.decode_c(latent)


class Vec2VecTrainer:
    """Trainer for Vec2Vec model"""
    
    def __init__(self, model: CodeVec2Vec, learning_rate: float = 0.001):
        self.model = model
        self.optimizer_gen = torch.optim.Adam(
            list(model.adapter_py.parameters()) + 
            list(model.adapter_c.parameters()) + 
            list(model.backbone.parameters()) + 
            list(model.output_py.parameters()) + 
            list(model.output_c.parameters()),
            lr=learning_rate
        )
        self.optimizer_disc = torch.optim.Adam(
            list(model.disc_py.parameters()) + 
            list(model.disc_c.parameters()) + 
            list(model.disc_latent_py.parameters()) + 
            list(model.disc_latent_c.parameters()),
            lr=learning_rate
        )
        
    def compute_losses(self, py_embeddings, c_embeddings, loss_weights=None):
        """Compute all losses for training"""
        
        # Forward passes
        py_latent = self.model.encode_python(py_embeddings)
        c_latent = self.model.encode_c(c_embeddings)
        
        py_reconstructed = self.model.decode_python(py_latent)
        c_reconstructed = self.model.decode_c(c_latent)
        
        py_to_c = self.model.translate_py_to_c(py_embeddings)
        c_to_py = self.model.translate_c_to_py(c_embeddings)
        
        # Cycle consistency
        py_cycled = self.model.translate_c_to_py(py_to_c)
        c_cycled = self.model.translate_py_to_c(c_to_py)
        
        # Reconstruction loss
        loss_rec_py = F.mse_loss(py_reconstructed, py_embeddings)
        loss_rec_c = F.mse_loss(c_reconstructed, c_embeddings)
        loss_rec = loss_rec_py + loss_rec_c
        
        # Cycle consistency loss
        loss_cycle_py = F.mse_loss(py_cycled, py_embeddings)
        loss_cycle_c = F.mse_loss(c_cycled, c_embeddings)
        loss_cycle = loss_cycle_py + loss_cycle_c
        
        # Vector Space Preservation (VSP) loss
        def vsp_loss(original, translated):
            orig_sim = F.cosine_similarity(original.unsqueeze(1), original.unsqueeze(0), dim=2)
            trans_sim = F.cosine_similarity(translated.unsqueeze(1), translated.unsqueeze(0), dim=2)
            return F.mse_loss(orig_sim, trans_sim)
        
        loss_vsp = vsp_loss(py_embeddings, py_to_c) + vsp_loss(c_embeddings, c_to_py)
        
        # Adversarial losses
        real_py_score = self.model.disc_py(py_embeddings)
        fake_py_score = self.model.disc_py(c_to_py)
        
        real_c_score = self.model.disc_c(c_embeddings)
        fake_c_score = self.model.disc_c(py_to_c)
        
        # Generator loss (wants to fool discriminator)
        loss_adv_gen = F.binary_cross_entropy(fake_py_score, torch.ones_like(fake_py_score)) + \
                       F.binary_cross_entropy(fake_c_score, torch.ones_like(fake_c_score))
        
        # Discriminator loss
        loss_adv_disc = F.binary_cross_entropy(real_py_score, torch.ones_like(real_py_score)) + \
                        F.binary_cross_entropy(fake_py_score, torch.zeros_like(fake_py_score)) + \
                        F.binary_cross_entropy(real_c_score, torch.ones_like(real_c_score)) + \
                        F.binary_cross_entropy(fake_c_score, torch.zeros_like(fake_c_score))
        
        # Total generator loss
        if loss_weights is None:
            loss_weights = {'rec': 0.5, 'cycle': 0.3, 'vsp': 0.1, 'adv': 0.1}

        loss_gen = loss_weights['rec'] * loss_rec + \
                   loss_weights['cycle'] * loss_cycle + \
                   loss_weights['vsp'] * loss_vsp + \
                   loss_weights['adv'] * loss_adv_gen
        
        # loss_gen = 0.5 * loss_rec + 0.3 * loss_cycle + 0.1 * loss_vsp + 0.1 * loss_adv_gen
        
        return {
            'loss_gen': loss_gen,
            'loss_disc': loss_adv_disc,
            'loss_rec': loss_rec,
            'loss_cycle': loss_cycle,
            'loss_vsp': loss_vsp,
            'loss_adv_gen': loss_adv_gen
        }
    
    def train_step(self, py_embeddings, c_embeddings, loss_weights=None):
        """Single training step"""
        
        if loss_weights is None:
            loss_weights = {'rec': 0.5, 'cycle': 0.3, 'vsp': 0.1, 'adv': 0.1}
        
        # Compute losses
        losses = self.compute_losses(py_embeddings, c_embeddings, loss_weights)
        
        # Update generator
        self.optimizer_gen.zero_grad()
        losses['loss_gen'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            list(self.model.adapter_py.parameters()) + 
            list(self.model.adapter_c.parameters()) + 
            list(self.model.backbone.parameters()) + 
            list(self.model.output_py.parameters()) + 
            list(self.model.output_c.parameters()),
            max_norm=1.0
        )
        self.optimizer_gen.step()
        
        # Update discriminator (recompute losses to avoid gradient issues)
        self.optimizer_disc.zero_grad()
        disc_losses = self.compute_discriminator_losses(py_embeddings, c_embeddings)
        disc_losses['loss_disc'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.disc_py.parameters()) + 
            list(self.model.disc_c.parameters()) + 
            list(self.model.disc_latent_py.parameters()) + 
            list(self.model.disc_latent_c.parameters()),
            max_norm=1.0
        )
        self.optimizer_disc.step()
        
        return {k: v.item() for k, v in {**losses, **disc_losses}.items()}
    
    def compute_discriminator_losses(self, py_embeddings, c_embeddings):
        """Compute discriminator losses separately to avoid gradient issues"""
        
        with torch.no_grad():
            py_to_c = self.model.translate_py_to_c(py_embeddings)
            c_to_py = self.model.translate_c_to_py(c_embeddings)
        
        # Recompute discriminator outputs
        real_py_score = self.model.disc_py(py_embeddings)
        fake_py_score = self.model.disc_py(c_to_py.detach())
        
        real_c_score = self.model.disc_c(c_embeddings)
        fake_c_score = self.model.disc_c(py_to_c.detach())
        
        # Discriminator loss
        loss_adv_disc = F.binary_cross_entropy(real_py_score, torch.ones_like(real_py_score)) + \
                        F.binary_cross_entropy(fake_py_score, torch.zeros_like(fake_py_score)) + \
                        F.binary_cross_entropy(real_c_score, torch.ones_like(real_c_score)) + \
                        F.binary_cross_entropy(fake_c_score, torch.zeros_like(fake_c_score))
        
        return {'loss_disc': loss_adv_disc}
