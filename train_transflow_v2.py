import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

print("=== Training TransFlow v2 - Improved Correlation ===")
sys.stdout.flush()

# 加载数据
X = np.load("data/human_ad/human_ad_X.npy").astype(np.float32)
labels = np.load("data/human_ad/human_ad_labels.npy", allow_pickle=True)
unique_labels = np.unique(labels)
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
y = np.array([label_to_idx[l] for l in labels])
n_classes = len(unique_labels)

print(f"Data: {X.shape}, Classes: {n_classes}")
print(f"Real data: mean={X.mean():.4f}, std={X.std():.4f}")
sys.stdout.flush()

device = "cuda"
input_dim = X.shape[1]

# 计算真实数据的基因均值
real_gene_mean = torch.FloatTensor(X.mean(axis=0)).to(device)

dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

class TransFlowModel(nn.Module):
    """Transformer-based Flow Matching Model"""
    def __init__(self, input_dim, n_classes, hidden_dim=512, n_heads=8, n_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 类别嵌入
        self.class_embed = nn.Embedding(n_classes, hidden_dim)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t, labels):
        # x: [B, D], t: [B, 1], labels: [B]
        h = self.input_proj(x)  # [B, H]
        t_emb = self.time_embed(t)  # [B, H]
        c_emb = self.class_embed(labels)  # [B, H]
        
        # 组合嵌入
        h = h + t_emb + c_emb
        h = h.unsqueeze(1)  # [B, 1, H]
        
        # Transformer
        h = self.transformer(h)
        h = h.squeeze(1)  # [B, H]
        
        # 输出
        return self.output_proj(h)

model = TransFlowModel(input_dim, n_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

def flow_matching_loss(model, x1, labels):
    """Optimal Transport Flow Matching Loss"""
    batch_size = x1.size(0)
    
    # 从标准正态分布采样x0
    x0 = torch.randn_like(x1)
    
    # 采样时间t
    t = torch.rand(batch_size, 1, device=device)
    
    # 线性插值: x_t = (1-t)*x0 + t*x1
    x_t = (1 - t) * x0 + t * x1
    
    # 目标速度场: v = x1 - x0
    target_v = x1 - x0
    
    # 预测速度场
    pred_v = model(x_t, t, labels)
    
    # MSE损失
    loss = F.mse_loss(pred_v, target_v)
    return loss

@torch.no_grad()
def generate_samples(model, n_samples, labels, n_steps=100):
    """使用ODE求解器生成样本"""
    model.eval()
    
    # 从标准正态分布开始
    x = torch.randn(n_samples, input_dim, device=device)
    
    # Euler方法积分
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples, 1), i * dt, device=device)
        v = model(x, t, labels)
        x = x + v * dt
    
    return x

print("Training 2000 epochs...")
sys.stdout.flush()
best_corr = -1
best_model_state = None

for epoch in range(2000):
    model.train()
    total_loss = 0
    for batch in loader:
        x, labels_batch = batch[0].to(device), batch[1].to(device)
        
        loss = flow_matching_loss(model, x, labels_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    
    if (epoch + 1) % 100 == 0:
        # 评估
        model.eval()
        with torch.no_grad():
            test_samples = []
            class_counts = np.bincount(y)
            class_probs = class_counts / class_counts.sum()
            n_per_class = (2000 * class_probs).astype(int)
            n_per_class[-1] = 2000 - n_per_class[:-1].sum()
            
            for c in range(n_classes):
                n = n_per_class[c]
                if n > 0:
                    labels_c = torch.full((n,), c, dtype=torch.long, device=device)
                    samples = generate_samples(model, n, labels_c, n_steps=50)
                    test_samples.append(samples)
            
            test_samples = torch.cat(test_samples, dim=0).cpu().numpy()
            corr = np.corrcoef(X.mean(0), test_samples.mean(0))[0, 1]
            
            if corr > best_corr:
                best_corr = corr
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/2000, Loss={avg_loss:.4f}, Corr={corr:.4f}, Best={best_corr:.4f}")
        sys.stdout.flush()

# 使用最佳模型生成最终样本
print(f"\nUsing best model with Corr={best_corr:.4f}")
sys.stdout.flush()

model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
model.eval()

with torch.no_grad():
    all_samples = []
    class_counts = np.bincount(y)
    class_probs = class_counts / class_counts.sum()
    n_per_class = (2000 * class_probs).astype(int)
    n_per_class[-1] = 2000 - n_per_class[:-1].sum()
    
    for c in range(n_classes):
        n = n_per_class[c]
        if n > 0:
            labels_c = torch.full((n,), c, dtype=torch.long, device=device)
            samples = generate_samples(model, n, labels_c, n_steps=100)
            all_samples.append(samples)
    
    transflow_samples = torch.cat(all_samples, dim=0).cpu().numpy()

np.random.shuffle(transflow_samples)
final_corr = np.corrcoef(X.mean(0), transflow_samples.mean(0))[0, 1]
print(f"Final TransFlow: mean={transflow_samples.mean():.4f}, std={transflow_samples.std():.4f}, Corr={final_corr:.4f}")

np.save("experiments/results/human_ad/transflow_samples.npy", transflow_samples)
torch.save(model.state_dict(), "experiments/results/human_ad/transflow_v2.pt")
print("Saved!")
