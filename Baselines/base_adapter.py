"""
Base adapter interface for baseline methods
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseAdapter(ABC):
    def __init__(self, input_dim: int, **kwargs):
        self.input_dim = input_dim
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(
        self, 
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def generate(
        self, 
        n_samples: int,
        **kwargs
    ) -> np.ndarray:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__.replace('Adapter', '')
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data
    
    def postprocess_samples(self, samples: np.ndarray) -> np.ndarray:
        return samples


class DummyAdapter(BaseAdapter):
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.mean = None
        self.std = None
    
    def train(
        self, 
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        **kwargs
    ) -> Dict[str, Any]:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.is_trained = True
        
        return {
            'final_loss': 0.0,
            'history': []
        }
    
    def generate(
        self, 
        n_samples: int,
        **kwargs
    ) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")
        
        samples = np.random.randn(n_samples, self.input_dim)
        samples = samples * self.std + self.mean
        
        return samples
    
    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)
    
    def load(self, path: str) -> None:
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        self.is_trained = True

