"""
scDiffusion adapter
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from scdiffusion_official_impl import scDiffusion

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_adapter import BaseAdapter


class scDiffusionAdapter(BaseAdapter):
    def __init__(self, input_dim: int, n_steps: int = 1000, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.n_steps = n_steps
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        hidden_dim = kwargs.get('hidden_dim', 512)
        self.model = scDiffusion(input_dim=input_dim, hidden_dim=hidden_dim, timesteps=n_steps, device=device)
        print(f"scDiffusion initialized: input_dim={input_dim}, timesteps={n_steps}")
    
    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 128, conditions: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        history = self.model.fit(data, epochs=epochs, batch_size=batch_size, verbose=kwargs.get('verbose', True))
        self.is_trained = True
        return history
    
    def generate(self, n_samples: int, conditions: Optional[np.ndarray] = None, guidance_scale: float = 1.0, **kwargs) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.generate(n_samples)
    
    def save(self, path: str) -> None:
        self.model.save(path)
    
    def load(self, path: str) -> None:
        self.model.load(path)
        self.is_trained = True


if __name__ == '__main__':
    adapter = scDiffusionAdapter(input_dim=2000, n_steps=1000)

