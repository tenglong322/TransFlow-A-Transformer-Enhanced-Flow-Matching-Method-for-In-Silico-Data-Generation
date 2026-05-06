
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from .flow_matching import FlowMatchingModel


class PositionalEncoding(nn.Module):

    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = query.shape[:2]
        

        residual = query
        

        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        #  [batch_size, num_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        

        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        

        output = self.w_o(attention_output)
        

        output = self.layer_norm(output + residual)
        
        return output
    
    def _scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) * self.temperature)
        

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        
        return output


class TransformerBlock(nn.Module):

    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU()
        }
        return activations.get(activation, nn.GELU())
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        attn_output = self.attention(x, x, x, mask)
        

        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + attn_output)
        
        return output


class TransformerVelocityField(nn.Module):

    
    def __init__(
        self,
        gene_dim: int,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_genes: int = 10000,
        time_embed_dim: int = 128,
        condition_dim: int = 0,
        dropout: float = 0.1,
        use_gene_embedding: bool = True
    ):
        super().__init__()
        
        self.gene_dim = gene_dim
        self.d_model = d_model
        self.use_gene_embedding = use_gene_embedding
        

        if use_gene_embedding:
            self.gene_embedding = nn.Linear(1, d_model)
            self.gene_projection = nn.Linear(gene_dim * d_model, d_model * gene_dim // 4)
        else:
            self.input_projection = nn.Linear(gene_dim, d_model)
        

        self.pos_encoding = PositionalEncoding(d_model, max_genes)
        

        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim // 2, time_embed_dim)
        )
        

        if condition_dim > 0:
            self.condition_projection = nn.Linear(condition_dim, d_model)
        

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        gene_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        batch_size, gene_dim = x.shape
        
        if self.use_gene_embedding:

            x_reshaped = x.unsqueeze(-1)  # [batch_size, gene_dim, 1]
            gene_embeds = self.gene_embedding(x_reshaped)  # [batch_size, gene_dim, d_model]
        else:

            x_projected = self.input_projection(x)  # [batch_size, d_model]
            gene_embeds = x_projected.unsqueeze(1).expand(-1, gene_dim, -1)
        

        gene_embeds = self.pos_encoding(gene_embeds)
        

        t_embed = self.time_embedding(t)  # [batch_size, time_embed_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, gene_dim, -1)  # [batch_size, gene_dim, time_embed_dim]
        

        if t_embed.shape[-1] != self.d_model:
            t_proj = nn.Linear(t_embed.shape[-1], self.d_model).to(x.device)
            t_embed = t_proj(t_embed)
        

        features = gene_embeds + t_embed
        

        if condition is not None:
            cond_embed = self.condition_projection(condition)
            cond_embed = cond_embed.unsqueeze(1).expand(-1, gene_dim, -1)
            features = features + cond_embed
        
        features = self.dropout(features)
        

        for layer in self.transformer_layers:
            features = layer(features, gene_mask)
        

        velocity = self.output_projection(features)  # [batch_size, gene_dim, 1]
        velocity = velocity.squeeze(-1)  # [batch_size, gene_dim]
        
        return velocity


class Transformerching(FlowMatchingModel):

    
    def __init__(
        self,
        gene_dim: int,
        transformer_config: Dict[str, Any],
        sigma_min: float = 1e-5,
        sigma_max: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_gene_attention_loss: bool = True
    ):
        
            velocity_network_config = {
            "hidden_dims": [512, 512],              "time_embed_dim": 128,
            "condition_dim": transformer_config.get("condition_dim", 0),
            "dropout": transformer_config.get("dropout", 0.1)
        }
        

        num_cell_types = transformer_config.get("condition_dim", 3)
        
        super().__init__(gene_dim, velocity_network_config, sigma_min, sigma_max, 
                        use_ema=use_ema, ema_decay=ema_decay, num_cell_types=num_cell_types)
        
        self.use_gene_attention_loss = use_gene_attention_loss
        

        self.velocity_net = TransformerVelocityField(gene_dim, **transformer_config)
        

        if use_ema:
            self.velocity_net_ema = TransformerVelocityField(gene_dim, **transformer_config)
            self.ema_decay = ema_decay
            self._copy_params_to_ema()
        else:
            self.velocity_net_ema = None
    
    def compute_gene_attention_loss(
        self, 
        velocity: torch.Tensor, 
        x0: torch.Tensor,
        gene_importance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if gene_importance is None:

            gene_importance = torch.mean(torch.abs(x0), dim=0)
            gene_importance = gene_importance / (torch.sum(gene_importance) + 1e-8)
        

        velocity_weights = torch.mean(torch.abs(velocity), dim=0)
        velocity_weights = velocity_weights / (torch.sum(velocity_weights) + 1e-8)
        

        kl_loss = F.kl_div(
            torch.log(velocity_weights + 1e-8),
            gene_importance,
            reduction='batchmean'
        )
        
        return kl_loss
    
    def compute_loss(
        self, 
        x0: torch.Tensor, 
        condition: Optional[torch.Tensor] = None,
        gene_mask: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> Dict[str, torch.Tensor]:

        
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
        

        predicted_velocity = self.velocity_net(x_t, t, condition, gene_mask)

        target_velocity = self.compute_target_velocity(x0, noise, t)
        

        if loss_type == "mse":
            fm_loss = F.mse_loss(predicted_velocity, target_velocity)
        elif loss_type == "l1":
            fm_loss = F.l1_loss(predicted_velocity, target_velocity)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        bio_loss = self._compute_biological_constraints(predicted_velocity, x0)
        
        attention_loss = torch.tensor(0.0, device=device)
        if self.use_gene_attention_loss:
            attention_loss = self.compute_gene_attention_loss(predicted_velocity, x0)
        
        total_loss = fm_loss + 0.1 * bio_loss + 0.05 * attention_loss
        
        return {
            "total_loss": total_loss,
            "fm_loss": fm_loss,
            "bio_loss": bio_loss,
            "attention_loss": attention_loss
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        condition: Optional[torch.Tensor] = None,
        gene_mask: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        method: str = "euler",
        temperature: float = 1.0
    ) -> torch.Tensor:
        
        model = self.velocity_net_ema if self.velocity_net_ema is not None else self.velocity_net
        model.eval()
        
        x = torch.randn(batch_size, self.gene_dim, device=device) * temperature
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            
            velocity = model(x, t, condition, gene_mask)
            
            if method == "euler":
                x = x + velocity * dt
            elif method == "heun":
                k1 = velocity
                x_tmp = x + k1 * dt
                t_next = torch.full((batch_size, 1), (i + 1) * dt, device=device)
                k2 = model(x_tmp, t_next, condition, gene_mask)
                x = x + (k1 + k2) * dt / 2
        
        return x
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:

        model = self.velocity_net_ema if self.velocity_net_ema is not None else self.velocity_net
        model.eval()
        
        with torch.no_grad():
            batch_size, gene_dim = x.shape
            

            if model.use_gene_embedding:
                x_reshaped = x.unsqueeze(-1)
                gene_embeds = model.gene_embedding(x_reshaped)
            else:
                x_projected = model.input_projection(x)
                gene_embeds = x_projected.unsqueeze(1).expand(-1, gene_dim, -1)
            

            gene_embeds = model.pos_encoding(gene_embeds)
            

            t_embed = model.time_embedding(t)
            t_embed = t_embed.unsqueeze(1).expand(-1, gene_dim, -1)
            
            if t_embed.shape[-1] != model.d_model:
                t_proj = nn.Linear(t_embed.shape[-1], model.d_model).to(x.device)
                t_embed = t_proj(t_embed)
            
            features = gene_embeds + t_embed
            
            if condition is not None:
                cond_embed = model.condition_projection(condition)
                cond_embed = cond_embed.unsqueeze(1).expand(-1, gene_dim, -1)
                features = features + cond_embed
            

            target_layer = model.transformer_layers[layer_idx]
            

            Q = target_layer.attention.w_q(features)
            K = target_layer.attention.w_k(features)
            
            batch_size, seq_len, d_model = Q.shape
            num_heads = target_layer.attention.num_heads
            d_k = d_model // num_heads
            
            Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
            K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            attention_weights = F.softmax(scores, dim=-1)
            

            attention_weights = torch.mean(attention_weights, dim=1)
            
        return attention_weights
