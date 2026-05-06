
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class VelocityField(nn.Module):

    
    def __init__(
        self,
        data_dim: int,
        hidden_dims: list = [512, 512, 512],
        time_embed_dim: int = 128,
        condition_dim: int = 0,
        dropout: float = 0.1,
        activation: str = "swish"
    ):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim
        

        self.time_embedding = self._build_time_embedding(time_embed_dim)
        

        input_dim = data_dim + time_embed_dim + condition_dim
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        

        layers.append(nn.Linear(prev_dim, data_dim))
        
        self.network = nn.Sequential(*layers)
        

        self._initialize_weights()
    
    def _build_time_embedding(self, embed_dim: int) -> nn.Module:

        return nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:

        activations = {
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh()
        }
        return activations.get(activation, nn.SiLU())
    
    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        Args:
            x:  [batch_size, data_dim]
            t:  [batch_size, 1]
            condition:  [batch_size, condition_dim]
        """
        batch_size = x.shape[0]
        

        t_embed = self.time_embedding(t)
        

        inputs = [x, t_embed]
        if condition is not None:
            inputs.append(condition)
        
        network_input = torch.cat(inputs, dim=-1)
        

        velocity = self.network(network_input)
        
        return velocity


class FlowMatchingModel(nn.Module):

    
    def __init__(
        self,
        data_dim: int,
        velocity_network_config: Dict[str, Any],
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        num_cell_types: int = 3
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.gene_dim = data_dim  
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_cell_types = num_cell_types
        

        self.velocity_net = VelocityField(data_dim, **velocity_network_config)
        

        if use_ema:
            self.velocity_net_ema = VelocityField(data_dim, **velocity_network_config)
            self.ema_decay = ema_decay
            self._copy_params_to_ema()
        else:
            self.velocity_net_ema = None
    
    def _copy_params_to_ema(self):

        if self.velocity_net_ema is not None:
            for param_main, param_ema in zip(
                self.velocity_net.parameters(), 
                self.velocity_net_ema.parameters()
            ):
                param_ema.data.copy_(param_main.data)
    
    def update_ema(self):

        if self.velocity_net_ema is not None:
            with torch.no_grad():
                for param_main, param_ema in zip(
                    self.velocity_net.parameters(), 
                    self.velocity_net_ema.parameters()
                ):
                    param_ema.data.mul_(self.ema_decay).add_(
                        param_main.data, alpha=1.0 - self.ema_decay
                    )
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:

        return torch.rand(batch_size, 1, device=device)
    
    def add_noise(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if noise is None:
            noise = torch.randn_like(x0)
        

        sigma_t = self._get_sigma(t)
        

        x_t = (1 - t) * noise + t * x0
        
        return x_t, noise
    
    def _get_sigma(self, t: torch.Tensor) -> torch.Tensor:

        return self.sigma_min + (self.sigma_max - self.sigma_min) * t
    
    def compute_target_velocity(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:

        return x0 - x1
    
    def forward(
        self, 
        x0: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:

        batch_size = x0.shape[0]
        device = x0.device
        

        if condition is not None:
            if condition.dim() == 1:  
                num_classes = getattr(self, 'num_cell_types', condition.max().item() + 1)
                condition_onehot = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float)
                condition_onehot.scatter_(1, condition.unsqueeze(1), 1)
                condition = condition_onehot
        

        t = self.sample_time(batch_size, device)
        noise = torch.randn_like(x0)
        

        x_t, _ = self.add_noise(x0, t, noise)
        

        predicted_velocity = self.velocity_net(x_t, t, condition)
        

        target_velocity = self.compute_target_velocity(x0, noise, t)
        
        if return_dict:
            return {
                "predicted_velocity": predicted_velocity,
                "target_velocity": target_velocity,
                "x_t": x_t,
                "t": t
            }
        
        return predicted_velocity, target_velocity
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        condition: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        method: str = "euler"
    ) -> torch.Tensor:


        if condition is not None:
            if condition.dim() == 1:                 
                num_classes = getattr(self, 'num_cell_types', condition.max().item() + 1)
                condition_onehot = torch.zeros(batch_size, num_classes, device=device, dtype=torch.float)
                condition_onehot.scatter_(1, condition.unsqueeze(1), 1)
                condition = condition_onehot
        

        model = self.velocity_net_ema if self.velocity_net_ema is not None else self.velocity_net
        model.eval()
        

        x = torch.randn(batch_size, self.gene_dim, device=device)
        
 
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            

            velocity = model(x, t, condition)
            
 
            if method == "euler":
                x = x + velocity * dt
            elif method == "heun":

                k1 = velocity
                x_tmp = x + k1 * dt
                t_next = torch.full((batch_size, 1), (i + 1) * dt, device=device)
                k2 = model(x_tmp, t_next, condition)
                x = x + (k1 + k2) * dt / 2
        
        return x
    
    def compute_loss(
        self, 
        x0: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> Dict[str, torch.Tensor]:

        
        predicted_velocity, target_velocity = self.forward(x0, condition)
        

        if loss_type == "mse":
            fm_loss = F.mse_loss(predicted_velocity, target_velocity)
        elif loss_type == "l1":
            fm_loss = F.l1_loss(predicted_velocity, target_velocity)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        bio_loss = self._compute_biological_constraints(predicted_velocity, x0)
        
        total_loss = fm_loss + 0.1 * bio_loss
        
        return {
            "total_loss": total_loss,
            "fm_loss": fm_loss,
            "bio_loss": bio_loss
        }
    
    def _compute_biological_constraints(
        self, 
        velocity: torch.Tensor, 
        x0: torch.Tensor
    ) -> torch.Tensor:

        sparsity_loss = torch.mean(torch.abs(velocity))
        

        negative_penalty = torch.mean(F.relu(-velocity))
        
        return sparsity_loss + negative_penalty

