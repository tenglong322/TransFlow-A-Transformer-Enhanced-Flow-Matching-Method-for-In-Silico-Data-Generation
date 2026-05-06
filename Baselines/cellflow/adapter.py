"""
cellFLOW adapter
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from cellflow_official_impl import cellFLOW

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_adapter import BaseAdapter


class cellFLOWAdapter(BaseAdapter):
    
    def __init__(
        self, 
        input_dim: int,
        use_ot: bool = True,
        **kwargs
    ):
        super().__init__(input_dim, **kwargs)
        self.use_ot = use_ot
        
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        hidden_dim = kwargs.get('hidden_dim', 512)
        
        self.model = cellFLOW(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        print(f"cellFLOW initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def train(
        self, 
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        
        history = self.model.fit(
            data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=kwargs.get('verbose', True)
        )
        
        self.is_trained = True
        return history
    
    def generate(
        self, 
        n_samples: int,
        use_ode_solver: bool = True,
        **kwargs
    ) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        n_steps = kwargs.get('n_steps', 100)
        samples = self.model.generate(n_samples, n_steps=n_steps)
        return samples
    
    def save(self, path: str) -> None:
        self.model.save(path)
    
    def load(self, path: str) -> None:
        self.model.load(path)
        self.is_trained = True


if __name__ == '__main__':
    adapter = cellFLOWAdapter(input_dim=2000, use_ot=True)

