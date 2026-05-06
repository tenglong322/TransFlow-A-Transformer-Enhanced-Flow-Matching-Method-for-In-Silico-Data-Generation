"""
scDiffusion implementation based on https://github.com/openai/guided-diffusion
DDPM with U-Net architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from tqdm import tqdm


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after the computation.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_head_channels = channels // num_heads
        else:
            self.num_head_channels = num_head_channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return h + x_in


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, use_conv=False, dims=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.dims = dims

        if dims == 1:
            conv_nd = nn.Conv1d
        else:
            raise NotImplementedError()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            conv_nd(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.0),
            conv_nd(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).unsqueeze(-1)
        h = h + emb_out
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class TimestepEmbedSequential(nn.Sequential):

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.0,
        channel_mult=(1, 2, 4),
        num_heads=4,
        use_checkpoint=False,
        dims=1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.dims = dims

        if dims == 1:
            conv_nd = nn.Conv1d
        else:
            raise NotImplementedError()

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = in_channels
        input_block_chans = []
        self.input_blocks = nn.ModuleList(
            [conv_nd(in_channels, model_channels, 3, padding=1)]
        )
        ch = model_channels
        input_block_chans.append(ch)
        
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, ch, dims=dims, use_checkpoint=use_checkpoint),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, ch, dims=dims, use_checkpoint=use_checkpoint),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            conv_nd(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        assert x.shape[0] == t.shape[0], f"batch size mismatch: {x.shape[0]} vs {t.shape[0]}"

        emb = self.time_embed(t.float())

        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv1d):
                h = module(h)
            else:
                h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)


class scDiffusion:
    def __init__(
        self,
        input_dim=2000,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(16,),
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        self.device = device
        self.input_dim = input_dim
        self.timesteps = timesteps

        self.model = UNetModel(
            in_channels=1,
            model_channels=model_channels,
            out_channels=1,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=(1, 2, 4),
            num_heads=4,
            dims=1,
        ).to(device)

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.losses = []

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        while len(sqrt_alphas_cumprod_t.shape) < len(x_0.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_noisy = self.q_sample(x_0, t, noise)
        
        x_noisy_reshaped = x_noisy.unsqueeze(1)  # [B, 1, input_dim]
        predicted_noise = self.model(x_noisy_reshaped, t).squeeze(1)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def fit(self, data, conditions=None, epochs=100, batch_size=32, verbose=True):
        data_tensor = torch.from_numpy(data).float().to(self.device)
        dataset = TensorDataset(data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(loader, disable=not verbose) if verbose else loader
            
            for batch in pbar:
                x_0 = batch[0]
                batch_size_curr = x_0.shape[0]
                
                t = torch.randint(0, self.timesteps, (batch_size_curr,)).to(self.device)
                
                loss = self.p_losses(x_0, t)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if verbose:
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(loader)
            self.losses.append(avg_loss)
            self.scheduler.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def generate(self, n_samples=100, batch_size=32):
        self.model.eval()
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        all_samples = []
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
            
            x = torch.randn(current_batch_size, 1, self.input_dim).to(self.device)
            
            for t in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps, desc=f"Sampling batch {batch_idx+1}/{num_batches}"):
                t_batch = torch.full((current_batch_size,), t, dtype=torch.long).to(self.device)
                
                predicted_noise = self.model(x, t_batch)
                
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                    x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
                else:
                    x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
                
                if t > 0 and t % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            all_samples.append(x.squeeze(1).cpu().numpy())
            
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        samples = np.vstack(all_samples)
        
        nan_mask = np.isnan(samples)
        if nan_mask.any():
            samples = np.nan_to_num(samples, nan=0.0)
        
        return samples

    def save(self, path):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'timesteps': self.timesteps,
            }
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

