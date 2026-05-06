"""
scGAN implementation based on https://github.com/imsb-uke/scGAN
WGAN-GP with Spectral Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from tqdm import tqdm


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Generator(nn.Module):
    def __init__(
        self, 
        latent_dim=128, 
        output_dim=2000, 
        num_conditions=0,
        condition_dim=32,
        hidden_dims=[512, 1024, 1024, 512]
    ):
        super().__init__()
        
        self.num_conditions = num_conditions
        self.use_conditioning = num_conditions > 0
        
        if self.use_conditioning:
            self.condition_embed = nn.Embedding(num_conditions, condition_dim)
            in_dim = latent_dim + condition_dim
        else:
            in_dim = latent_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Softplus())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z, conditions=None):
        if self.use_conditioning and conditions is not None:
            c_emb = self.condition_embed(conditions)
            z = torch.cat([z, c_emb], dim=1)
        
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(
        self, 
        input_dim=2000, 
        num_conditions=0,
        condition_dim=32,
        hidden_dims=[512, 256, 128]
    ):
        super().__init__()
        
        self.num_conditions = num_conditions
        self.use_conditioning = num_conditions > 0
        
        if self.use_conditioning:
            self.condition_embed = nn.Embedding(num_conditions, condition_dim)
            in_dim = input_dim + condition_dim
        else:
            in_dim = input_dim
        
        layers = []
        for hidden_dim in hidden_dims:
            linear = nn.Linear(in_dim, hidden_dim)
            layers.append(nn.utils.spectral_norm(linear))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.utils.spectral_norm(nn.Linear(in_dim, 1))
    
    def forward(self, x, conditions=None):
        if self.use_conditioning and conditions is not None:
            c_emb = self.condition_embed(conditions)
            x = torch.cat([x, c_emb], dim=1)
        
        features = self.features(x)
        return self.output(features), features


class scGAN:
    def __init__(
        self,
        input_dim=2000,
        latent_dim=128,
        num_conditions=0,
        condition_dim=32,
        hidden_dims_g=[512, 1024, 1024, 512],
        hidden_dims_d=[512, 256, 128],
        device='cuda'
    ):
        self.device = device
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        self.use_conditioning = num_conditions > 0
        
        self.generator = Generator(
            latent_dim, input_dim, num_conditions, condition_dim, hidden_dims_g
        ).to(device)
        self.discriminator = Discriminator(
            input_dim, num_conditions, condition_dim, hidden_dims_d
        ).to(device)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    def compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        d_interpolates, _ = self.discriminator(interpolates)
        
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data, conditions=None, n_critic=5, lambda_gp=10.0):
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        if conditions is not None:
            conditions = conditions.to(self.device)
        
        for _ in range(n_critic):
            self.d_optimizer.zero_grad()
            
            real_validity, real_features = self.discriminator(real_data, conditions)
            
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z, conditions).detach()
            fake_validity, _ = self.discriminator(fake_data, conditions)
            
            gp = self.compute_gradient_penalty(real_data, fake_data)
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
            
            d_loss.backward()
            self.d_optimizer.step()
        
        self.g_optimizer.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.generator(z, conditions)
        fake_validity, fake_features = self.discriminator(fake_data, conditions)
        
        g_loss = -torch.mean(fake_validity)
        _, real_features = self.discriminator(real_data, conditions)
        feature_loss = F.mse_loss(fake_features.mean(0), real_features.mean(0))
        g_loss += feature_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'gp': gp.item(),
            'real_score': real_validity.mean().item(),
            'fake_score': fake_validity.mean().item()
        }
    
    def fit(self, data, conditions=None, epochs=100, batch_size=128, verbose=True):
        if conditions is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(data),
                torch.LongTensor(conditions)
            )
        else:
            dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {'d_loss': [], 'g_loss': []}
        
        pbar_epoch = tqdm(range(epochs), desc="Training scGAN") if verbose else range(epochs)
        
        for epoch in pbar_epoch:
            epoch_d_loss = []
            epoch_g_loss = []
            
            pbar_batch = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if verbose else dataloader
            
            for batch in pbar_batch:
                if conditions is not None:
                    batch_data, batch_conditions = batch
                    metrics = self.train_step(batch_data, batch_conditions)
                else:
                    batch_data = batch[0]
                    metrics = self.train_step(batch_data, None)
                
                epoch_d_loss.append(metrics['d_loss'])
                epoch_g_loss.append(metrics['g_loss'])
                
                if verbose and isinstance(pbar_batch, tqdm):
                    pbar_batch.set_postfix({
                        'd_loss': f"{metrics['d_loss']:.4f}",
                        'g_loss': f"{metrics['g_loss']:.4f}"
                    })
            
            avg_d_loss = np.mean(epoch_d_loss)
            avg_g_loss = np.mean(epoch_g_loss)
            
            history['d_loss'].append(avg_d_loss)
            history['g_loss'].append(avg_g_loss)
            
            if verbose and isinstance(pbar_epoch, tqdm):
                pbar_epoch.set_postfix({
                    'D_loss': f"{avg_d_loss:.4f}",
                    'G_loss': f"{avg_g_loss:.4f}"
                })
        
        return history
    
    def generate(self, n_samples=1000, conditions=None, batch_size=500):
        self.generator.eval()
        
        all_samples = []
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Generating scGAN samples", total=num_batches):
                current_batch_size = min(batch_size, n_samples - i)
                z = torch.randn(current_batch_size, self.latent_dim).to(self.device)
                
                if conditions is not None:
                    batch_conditions = torch.LongTensor(conditions[i:i+current_batch_size]).to(self.device)
                    fake_data = self.generator(z, batch_conditions)
                else:
                    fake_data = self.generator(z, None)
                
                all_samples.append(fake_data.cpu().numpy())
        
        return np.vstack(all_samples)
    
    def save(self, path):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

