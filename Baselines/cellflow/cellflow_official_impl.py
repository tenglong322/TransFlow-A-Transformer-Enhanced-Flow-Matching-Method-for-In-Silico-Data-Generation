"""
CellFlow implementation based on https://github.com/theislab/CellFlow
OT-CFM (Optimal Transport Conditional Flow Matching)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def sinkhorn_algorithm(C, epsilon=0.1, num_iters=100):
    n, m = C.shape
    
    K = torch.exp(-C / epsilon)
    u = torch.ones(n, device=C.device) / n
    v = torch.ones(m, device=C.device) / m
    
    for _ in range(num_iters):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.t() @ u)
    
    P = torch.diag(u) @ K @ torch.diag(v)
    
    return P


def compute_optimal_transport_plan(x_0, x_1, epsilon=0.1, num_iters=100):
    batch_size = x_0.size(0)
    
    C = torch.cdist(x_0, x_1, p=2) ** 2
    P = sinkhorn_algorithm(C, epsilon=epsilon, num_iters=num_iters)
    indices = torch.argmax(P, dim=1)
    x_1_matched = x_1[indices]
    
    return x_1_matched


class VelocityField(nn.Module):
    def __init__(self, input_dim=2000, hidden_dim=512, time_emb_dim=64):
        super().__init__()
        
        self.time_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, input_dim)
        )
    
    def time_embedding(self, t):
        half_dim = self.time_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = torch.cat([x, t_emb], dim=-1)
        v = self.net(h)
        
        return v


class cellFLOW:
    def __init__(
        self,
        input_dim=2000,
        hidden_dim=512,
        time_emb_dim=64,
        ot_epsilon=0.1,
        ot_iters=100,
        device='cuda'
    ):
        self.device = device
        self.input_dim = input_dim
        self.ot_epsilon = ot_epsilon
        self.ot_iters = ot_iters
        self.velocity_net = VelocityField(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.velocity_net.parameters(),
            lr=2e-4,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
    
    def sample_ot_plan(self, x_0, x_1):
        x_1_matched = compute_optimal_transport_plan(
            x_0, x_1, 
            epsilon=self.ot_epsilon,
            num_iters=self.ot_iters
        )
        return x_1_matched
    
    def compute_conditional_flow(self, x_0, x_1, t):
        t = t.view(-1, 1)
        x_t = t * x_1 + (1 - t) * x_0
        return x_t
    
    def compute_target_velocity(self, x_0, x_1):
        return x_1 - x_0
    
    def train_step(self, batch):
        batch = batch.to(self.device)
        batch_size = batch.size(0)
        
        x_1 = batch
        x_0 = torch.randn_like(x_1)
        x_1_paired = self.sample_ot_plan(x_0, x_1)
        t = torch.rand(batch_size, device=self.device)
        x_t = self.compute_conditional_flow(x_0, x_1_paired, t)
        target_v = self.compute_target_velocity(x_0, x_1_paired)
        pred_v = self.velocity_net(x_t, t)
        loss = F.mse_loss(pred_v, target_v)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample_ode(self, x_0, n_steps=100, show_progress=True):
        x = x_0
        dt = 1.0 / n_steps
        
        iterator = tqdm(range(n_steps), desc="Solving ODE", leave=False) if show_progress else range(n_steps)
        
        for i in iterator:
            t = torch.full((x.size(0),), i / n_steps, device=self.device)
            v = self.velocity_net(x, t)
            x = x + v * dt
        
        return x
    
    def fit(self, data, epochs=100, batch_size=128, verbose=True):
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_losses = []
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else dataloader
            
            for batch_data, in pbar:
                loss = self.train_step(batch_data)
                epoch_losses.append(loss)
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            self.scheduler.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        return history
    
    def generate(self, n_samples=1000, n_steps=100, batch_size=500):
        self.velocity_net.eval()
        
        all_samples = []
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Generating cellFLOW samples", total=num_batches):
            current_batch_size = min(batch_size, n_samples - i)
            x_0 = torch.randn(current_batch_size, self.input_dim, device=self.device)
            samples = self.sample_ode(x_0, n_steps=n_steps, show_progress=(i==0))
            samples = torch.clamp(samples, min=0)
            
            all_samples.append(samples.cpu().numpy())
        
        return np.vstack(all_samples)
    
    def save(self, path):
        torch.save({
            'velocity_net': self.velocity_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.velocity_net.load_state_dict(checkpoint['velocity_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

