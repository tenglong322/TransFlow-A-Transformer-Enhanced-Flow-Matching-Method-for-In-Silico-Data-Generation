"""
scGAN adapter
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(__file__))
from scgan_official_impl import scGAN

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_adapter import BaseAdapter


class scGANAdapter(BaseAdapter):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim, **kwargs)
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        latent_dim = kwargs.get('latent_dim', 128)
        self.model = scGAN(input_dim=input_dim, latent_dim=latent_dim, device=device)
        print(f"scGAN initialized: input_dim={input_dim}, latent_dim={latent_dim}")
    
    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 128, **kwargs) -> Dict[str, Any]:
        history = self.model.fit(data, epochs=epochs, batch_size=batch_size, verbose=kwargs.get('verbose', True))
        self.is_trained = True
        return history
    
    def generate(self, n_samples: int, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.generate(n_samples)
    
    def save(self, path: str) -> None:
        self.model.save(path)
    
    def load(self, path: str) -> None:
        self.model.load(path)
        self.is_trained = True


if __name__ == '__main__':
    adapter = scGANAdapter(input_dim=2000)

