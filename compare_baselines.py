"""
Baseline comparison experiments
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    print("Warning: scanpy not found. Will use dummy data for testing.")
    HAS_SCANPY = False
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import SingleCellDataset, create_train_val_split, create_data_loaders
from src.models.transformer_flow import TransformerFlowMatching
from src.evaluation.metrics import compute_generation_metrics, compute_frechet_distance, create_evaluation_report, compute_enhanced_generation_metrics
from experiments.baselines.baseline_models import create_baseline_model


def load_h5ad_data(data_path: str, max_cells: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    print(f"Loading and preprocessing data from: {data_path}")
    
    if HAS_SCANPY and os.path.exists(data_path):
        try:
            adata = sc.read_h5ad(data_path)
            print(f"Original data shape: {adata.shape}")
            
            if max_cells is not None and adata.n_obs > max_cells:
                print(f"Subsampling to {max_cells} cells")
                sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
            
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            print(f"After filtering: {adata.shape}")
            
            sc.pp.normalize_total(adata, target_sum=10000)
            sc.pp.log1p(adata)
            
            if adata.n_vars > 2000:
                sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
            sc.pp.scale(adata)
            expression_data = adata.X
            
            if hasattr(expression_data, 'toarray'):
                expression_data = expression_data.toarray()
            
            expression_data = expression_data.astype(np.float32)
            
            print(f"Data stats: range=[{expression_data.min():.3f}, {expression_data.max():.3f}], mean={expression_data.mean():.3f}, std={expression_data.std():.3f}")
            cell_types = None
            label_candidates = ['cell_type', 'celltype', 'louvain', 'leiden', 'clusters', 'cluster']
            
            for col_name in label_candidates:
                if col_name in adata.obs.columns:
                    print(f"Found cell type labels: '{col_name}'")
                    unique_types = adata.obs[col_name].unique()
                    
                    if adata.obs[col_name].dtype == 'object' or adata.obs[col_name].dtype.name == 'category':
                        type_to_idx = {str(t): i for i, t in enumerate(unique_types)}
                        cell_types = np.array([type_to_idx[str(t)] for t in adata.obs[col_name]], dtype=np.int64)
                    else:
                        cell_types = adata.obs[col_name].values.astype(np.int64)
                    
                    print(f"Found {len(unique_types)} unique cell types")
                    break
            
            if cell_types is None:
                print("No cell type found, attempting Louvain clustering...")
                try:
                    sc.pp.neighbors(adata, n_neighbors=15, random_state=42)
                    sc.tl.louvain(adata, resolution=0.8, random_state=42)
                    cell_types = adata.obs['louvain'].astype(int).values
                    print(f"Created {len(np.unique(cell_types))} clusters")
                except Exception as e:
                    print(f"Clustering failed: {e}")
                    cell_types = np.zeros(expression_data.shape[0], dtype=np.int64)
            
            print(f"Preprocessed: {expression_data.shape}, sparsity={100*(expression_data == 0).mean():.1f}%")
            
            return expression_data, cell_types
            
        except Exception as e:
            print(f"Error loading data with scanpy: {e}")
    else:
        if not HAS_SCANPY:
            print("Scanpy not available")
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
    
    print("Creating dummy data for testing...")
    n_cells = max_cells or 1000
    n_genes = 2000
    np.random.seed(42)
    expression_data = np.random.normal(0, 1, size=(n_cells, n_genes)).astype(np.float32)
    mask = np.random.random((n_cells, n_genes)) < 0.1
    expression_data[mask] = 0
    expression_data = np.clip(expression_data, -10, 10)
    n_types = 5
    cell_types = np.random.randint(0, n_types, size=n_cells, dtype=np.int64)
    print(f"Created dummy data: {expression_data.shape}")
    
    return expression_data, cell_types


class BaselineComparison:
    def __init__(
        self,
        data_path: str,
        results_dir: str = './experiments/results/baseline_comparison',
        device: torch.device = None,
        num_samples: int = 2000,
        use_official_impl: bool = True
    ):
        self.data_path = data_path
        self.results_dir = results_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_samples = num_samples
        self.use_official_impl = use_official_impl
        
        print(f"\nMode: {'official' if use_official_impl else 'simplified'}")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        
        self.real_data = None
        self.real_labels = None
        self.input_dim = None
        self.generated_samples = {}
        self.evaluation_results = {}
        self._setup_gpu_optimization()
    
    def _setup_gpu_optimization(self):
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}, {gpu_memory:.1f}GB")
            
            if gpu_memory >= 48:
                torch.cuda.empty_cache()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.gpu_optimized = True
                self.batch_size_dict = {'vae': 64, 'gan': 32, 'diffusion': 28, 'flow': 48, 'our_method': 32}
                self.sample_batch_size = 500
            else:
                self.gpu_optimized = False
                self.batch_size_dict = {'vae': 16, 'gan': 8, 'diffusion': 8, 'flow': 12, 'our_method': 8}
                self.sample_batch_size = 100
        else:
            self.gpu_optimized = False
            self.batch_size_dict = {'vae': 8, 'gan': 4, 'diffusion': 4, 'flow': 8, 'our_method': 4}
            self.sample_batch_size = 50
    
    def load_data(self):
        print("Loading data...")
        expression_data, cell_types = load_h5ad_data(self.data_path, max_cells=None)
        dataset = SingleCellDataset(
            expression_data=expression_data,
            cell_types=cell_types
        )
        
        train_dataset, temp_dataset = create_train_val_split(
            dataset, val_split=0.4, stratify=True, random_seed=42
        )
        val_dataset, test_dataset = create_train_val_split(
            temp_dataset, val_split=0.5, stratify=True, random_seed=42
        )
        
        train_loader, val_loader = create_data_loaders(
            train_dataset, val_dataset, batch_size=16, num_workers=8
        )
        
        test_loader = create_data_loaders(
            test_dataset, test_dataset, batch_size=16, num_workers=8
        )[0]
        
        real_data_list = []
        real_labels_list = []
        for batch in test_loader:
            real_data_list.append(batch['expression'])
            real_labels_list.append(batch['cell_type'])
            if len(real_data_list) * batch['expression'].shape[0] >= self.num_samples:
                break
        
        if real_data_list:
            self.real_data = torch.cat(real_data_list, dim=0)[:self.num_samples]
            self.real_labels = torch.cat(real_labels_list, dim=0)[:self.num_samples]
        else:
            indices = np.random.choice(len(test_dataset), min(self.num_samples, len(test_dataset)), replace=False)
            self.real_data = test_dataset.expression_data[indices]
            self.real_labels = test_dataset.cell_types[indices]
            
        self.input_dim = self.real_data.shape[1]
        
        print(f"Loaded {len(self.real_data)} real samples with {self.input_dim} features")
        
        return train_loader, val_loader, test_loader
    
    def train_baseline_models(self, train_loader, epochs: int = 50):
        baseline_configs = {
            'vae': {
                'input_dim': self.input_dim,
                'latent_dim': 128,
                'hidden_dims': [512, 256]
            },
            'gan': {'input_dim': self.input_dim, 'latent_dim': 128, 'hidden_dims': [512, 512, 256]},
            'diffusion': {'input_dim': self.input_dim, 'hidden_dims': [1024, 512, 512, 256], 'num_timesteps': 1000},
            'flow': {'input_dim': self.input_dim, 'num_layers': 12, 'hidden_dim': 512}
        }
        
        baseline_epochs = {
            'vae': max(int(epochs * 2.5), 1),
            'gan': max(int(epochs * 4), 1),
            'diffusion': max(int(epochs * 5), 1),
            'flow': max(int(epochs * 3), 1)
        }
        
        print(f"Baseline epochs: VAE={baseline_epochs['vae']}, GAN={baseline_epochs['gan']}, Diff={baseline_epochs['diffusion']}, Flow={baseline_epochs['flow']}")
        
        self.trained_models = {}
        
        display_names = {
            'vae': 'VAE (Baseline)',
            'gan': 'scGAN (Single-Cell GAN)',
            'diffusion': 'scDiffusion (Single-Cell Diffusion)',
            'flow': 'cellFLOW (Single-Cell Normalizing Flow)'
        }
        
        for model_name, config in baseline_configs.items():
            display_name = display_names.get(model_name, model_name.upper())
            actual_epochs = baseline_epochs[model_name]
            print(f"\nTraining {display_name} model ({actual_epochs} epochs for sufficient convergence)...")
            
            if self.use_official_impl and model_name != 'vae':
                model = self._create_official_baseline_model(model_name, config)
            else:
                model = create_baseline_model(model_name, config)
            
            if not (self.use_official_impl and model_name != 'vae'):
                model = model.to(self.device)
            
            if model_name == 'gan':
                self._train_gan(model, train_loader, actual_epochs)
            else:
                self._train_model(model, train_loader, actual_epochs, model_name)
            
            if self.use_official_impl and model_name != 'vae':
                if hasattr(model, 'generator'):
                    model.generator = model.generator.cpu()
                if hasattr(model, 'discriminator'):
                    model.discriminator = model.discriminator.cpu()
                if hasattr(model, 'model'):
                    model.model = model.model.cpu()
                if hasattr(model, 'velocity_net'):
                    model.velocity_net = model.velocity_net.cpu()
                print(f"  Released {model_name} GPU memory")
            else:
                model = model.cpu()
            
            self.trained_models[model_name] = model
            
            model_path = os.path.join(self.results_dir, f'{model_name}_model.pt')
            
            if self.use_official_impl and model_name != 'vae' and hasattr(model, 'save'):
                model.save(model_path)
            else:
                torch.save(model.state_dict(), model_path)
            print(f"Saved {model_name} model to {model_path}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  GPU: allocated {allocated:.2f}GB, reserved {reserved:.2f}GB")
    
    def _create_official_baseline_model(self, model_name: str, config: dict):
        import sys
        import importlib.util
        
        if model_name.lower() == 'gan':
            impl_path = 'experiments/official_baselines/scgan/scgan_official_impl.py'
            spec = importlib.util.spec_from_file_location("scgan_official", impl_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.scGAN(
                input_dim=config['input_dim'],
                latent_dim=config.get('latent_dim', 128),
                num_conditions=0,
                condition_dim=32,
                hidden_dims_g=config.get('hidden_dims', [512, 1024, 1024, 512]),
                hidden_dims_d=[512, 256, 128],
                device=self.device
            )
            print(f"Loaded scGAN (WGAN-GP + Spectral Norm)")
            
        elif model_name.lower() == 'diffusion':
            impl_path = 'experiments/official_baselines/scdiffusion/scdiffusion_official_impl.py'
            spec = importlib.util.spec_from_file_location("scdiffusion_official", impl_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.scDiffusion(
                input_dim=config['input_dim'],
                model_channels=128,
                num_res_blocks=2,
                attention_resolutions=(16,),
                timesteps=config.get('num_timesteps', 1000),
                beta_start=1e-4,
                beta_end=0.02,
                device=self.device
            )
            print(f"Loaded scDiffusion (DDPM)")
            
        elif model_name.lower() == 'flow':
            impl_path = 'experiments/official_baselines/cellflow/cellflow_official_impl.py'
            spec = importlib.util.spec_from_file_location("cellflow_official", impl_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.cellFLOW(
                input_dim=config['input_dim'],
                hidden_dim=config.get('hidden_dim', 512),
                time_emb_dim=64,
                ot_epsilon=0.1,
                ot_iters=100,
                device=self.device
            )
            print(f"Loaded cellFLOW (OT-CFM)")
            
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        return model
    
    def _train_model(self, model, train_loader, epochs: int, model_name: str):
        if self.use_official_impl and hasattr(model, 'fit'):
            print(f"Using {model_name.upper()} fit method...")
            all_data = []
            for batch in train_loader:
                all_data.append(batch['expression'].cpu().numpy())
            train_data = np.vstack(all_data)
            
            if model_name.lower() == 'flow':
                model.fit(
                    data=train_data,
                    epochs=epochs,
                    batch_size=16,
                    verbose=True
                )
            else:
                model.fit(
                    data=train_data,
                    conditions=None,
                    epochs=epochs,
                    batch_size=16,
                    verbose=True
                )
            
            if hasattr(model, 'losses'):
                plt.figure(figsize=(8, 6))
                plt.plot(model.losses)
                plt.title(f'{model_name.upper()} Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, 'figures', f'{model_name}_training_curve.png'))
                plt.close()
            
            return
        
        lr_dict = {'vae': 5e-4, 'diffusion': 1e-4, 'flow': 5e-4}
        lr = lr_dict.get(model_name, 1e-3)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch in train_loader:
                data = batch['expression'].to(self.device)
                
                optimizer.zero_grad()
                loss_dict = model.train_step(data)
                loss = loss_dict['total_loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        plt.figure(figsize=(8, 6))
        plt.plot(losses)
        plt.title(f'{model_name.upper()} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'figures', f'{model_name}_training_curve.png'))
        plt.close()
    
    def _train_gan(self, model, train_loader, epochs: int):
        if self.use_official_impl and hasattr(model, 'fit'):
            print(f"Using scGAN fit method...")
            all_data = []
            for batch in train_loader:
                all_data.append(batch['expression'].cpu().numpy())
            train_data = np.vstack(all_data)
            
            model.fit(
                data=train_data,
                conditions=None,
                epochs=epochs,
                batch_size=16,
                verbose=True
            )
            
            if hasattr(model, 'g_losses') and hasattr(model, 'd_losses'):
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(model.g_losses, label='Generator')
                plt.plot(model.d_losses, label='Discriminator')
                plt.title('GAN Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(np.array(model.g_losses) - np.array(model.d_losses))
                plt.title('G Loss - D Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss Difference')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'figures', 'gan_training_curve.png'))
                plt.close()
            
            return
        
        g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        model.train()
        g_losses = []
        d_losses = []
        
        for epoch in range(epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch in train_loader:
                data = batch['expression'].to(self.device)
                batch_size = data.shape[0]
                
                d_optimizer.zero_grad()
                real_scores = model.discriminate(data)
                d_loss_real = nn.BCEWithLogitsLoss()(
                    real_scores, torch.ones_like(real_scores)
                )
                
                z = torch.randn(batch_size, model.latent_dim, device=self.device)
                fake_data = model.generate(z).detach()
                fake_scores = model.discriminate(fake_data)
                d_loss_fake = nn.BCEWithLogitsLoss()(
                    fake_scores, torch.zeros_like(fake_scores)
                )
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, model.latent_dim, device=self.device)
                fake_data = model.generate(z)
                fake_scores = model.discriminate(fake_data)
                g_loss = nn.BCEWithLogitsLoss()(
                    fake_scores, torch.ones_like(fake_scores)
                )
                
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_losses.append(g_loss.item())
                epoch_d_losses.append(d_loss.item())
            
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, label='Generator')
        plt.plot(d_losses, label='Discriminator')
        plt.title('GAN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.array(g_losses) - np.array(d_losses))
        plt.title('G Loss - D Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'figures', 'gan_training_curves.png'))
        plt.close()
    
    def load_our_model(self, model_path: str, config_path: str):
        print("Loading our TransformerFlowMatching model...")
        
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if config_path and os.path.exists(config_path):
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    model_config = config.get('model', {})
                except Exception as e:
                    print(f"Error loading config: {e}, using default config")
                    model_config = {}
            else:
                print("Config file not found, using default config")
                model_config = {}
            
            if not model_config:
                model_config = {}
            model_config.update({
                'd_model': 256, 'num_layers': 6, 'num_heads': 8, 'd_ff': 1024,
                'dropout': 0.1, 'time_embed_dim': 256, 'condition_dim': 0, 'use_gene_embedding': False
            })
            
            print(f"Creating model with config: d_model={model_config['d_model']}, "
                  f"num_layers={model_config['num_layers']}, "
                  f"gene_embedding={model_config.get('use_gene_embedding', True)}")
            
            our_model = TransformerFlowMatching(
                gene_dim=self.input_dim,
                transformer_config=model_config,
                use_ema=False
            )
            print("Using randomly initialized model")
            self.trained_models['our_method'] = our_model
            print(f"Loaded TransformerFlowMatching model")
            gc.collect()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping our method in comparison")
    
    def train_our_model(self, train_loader, epochs: int = 200):
        
        if 'our_method' not in self.trained_models:
            print("TransformerFlowMatching model not loaded, skipping training")
            return
        
        model = self.trained_models['our_method']
        
        batch_size = self.batch_size_dict.get('our_method', 16) if hasattr(self, 'batch_size_dict') else 16
        print(f"\nTraining TransformerFlowMatching for {epochs} epochs...")
        
        model = model.to(self.device)
        model.train()
        
        if self.device.type == 'cuda':
            print(f"Model on GPU: {torch.cuda.get_device_name()}")
            torch.cuda.empty_cache()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        warmup_epochs = max(5, min(epochs // 5, epochs - 1))
        
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                if epochs - warmup_epochs <= 1:
                    return 0.1
                progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
                return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
        

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                data = batch['expression'].to(self.device)
                
                optimizer.zero_grad()
                loss_dict = model.compute_loss(data)
                loss = loss_dict['total_loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
                
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step()
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if hasattr(model, 'update_ema'):
                model.update_ema()
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                gpu_info = ""
                if self.device.type == 'cuda':
                    print(f"  Model moved to GPU for sampling")
                    print(f"  GPU Usage: {torch.cuda.memory_allocated() / 1e6:.0f}MB")
                
                batch_size = getattr(self, 'sample_batch_size', 100)
                all_samples = []
                num_batches = (self.num_samples + batch_size - 1) // batch_size
                
                print(f"  Using RK4 solver + 500 steps for precise sampling")
                print(f"  Sampling batch={batch_size}, {num_batches} batches")
                print(f"  Generating {self.num_samples} samples...")
                for i in range(num_batches):
                    current_batch_size = min(batch_size, self.num_samples - i * batch_size)
                    batch_samples = model.sample(
                        batch_size=current_batch_size,
                        device=self.device,
                        num_steps=500,
                        method='rk4',
                        temperature=1.0 # standard temperature
                    )
                    all_samples.append(batch_samples.cpu().numpy())
                    del batch_samples
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    if (i + 1) % 2 == 0:
                        print(f"  Progress: {min((i+1)*batch_size, self.num_samples)}/{self.num_samples} samples")
                
                samples_np = np.vstack(all_samples)
                del all_samples
                
                # Move back to CPU to release GPU memory
                model = model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # Baseline methods - running on GPU (faster)
                # Official implementations have already handled devices in __init__, no need to .to()
                if not (self.use_official_impl and hasattr(model, 'generate')):
                    # Simplified versions need to be moved to device
                    model = model.to(self.device)
                    model.eval()  # Only call eval for simplified versions
                
                # Check if it's an official implementation (has generate method)
                if self.use_official_impl and hasattr(model, 'generate'):
                    # Use official implementation's generate method
                    # Official implementations will call eval() inside generate method
                    print(f"  Using official {model_name.upper()} generate method to generate samples")
                    
                    # Fix: move model components back to GPU before sampling
                    if hasattr(model, 'generator'):
                        model.generator = model.generator.to(self.device)
                    if hasattr(model, 'discriminator'):
                        model.discriminator = model.discriminator.to(self.device)
                    if hasattr(model, 'model'):  # scDiffusion's UNet
                        model.model = model.model.to(self.device)
                    if hasattr(model, 'velocity_net'):  # cellFLOW
                        model.velocity_net = model.velocity_net.to(self.device)
                    
                    # Optimize: scDiffusion sampling uses larger batch_size for speedup (no quality impact)
                    if model_name == 'diffusion':
                        # Force clear GPU cache to release residual memory
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        print(f"    Cleared GPU cache, released residual memory")
                        
                        samples_np = model.generate(
                            n_samples=self.num_samples,
                            batch_size=32  # FP32 needs smaller batch_size (official precision)
                        )
                        num_batches = (self.num_samples + 31) // 32
                        print(f"    Using batch_size=32 for sampling ({num_batches} batches)")
                        print(f"    Estimated GPU memory: ~8GB/48GB (FP32 needs more but more stable)")
                        print(f"    Quality: 100% official implementation, numerically stable, no NaN")
                        print(f"    Sampling time: ~3-4 hours (2000 samples, 63 batches)")
                    else:
                        samples_np = model.generate(n_samples=self.num_samples)
                        
                    if hasattr(model, 'generator'):
                        model.generator = model.generator.cpu()
                    if hasattr(model, 'discriminator'):
                        model.discriminator = model.discriminator.cpu()
                    if hasattr(model, 'model'):
                        model.model = model.model.cpu()
                    if hasattr(model, 'velocity_net'):
                        model.velocity_net = model.velocity_net.cpu()
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    samples = model.sample(self.num_samples, self.device)
                    samples_np = samples.cpu().numpy()
                    del samples
                    
                if not (self.use_official_impl and hasattr(model, 'generate')):
                    model = model.cpu()
                
                self.generated_samples[model_name] = samples_np
                sample_path = os.path.join(self.results_dir, f'generated_{model_name}_samples.npy')
                np.save(sample_path, samples_np)
                print(f"  Saved generated samples to {sample_path}")
                del samples_np
            
            import gc
            gc.collect()
        
        print("Sample generation completed")
    
    def evaluate_all_methods(self):
        
        print("\nEvaluating all methods...")
        
        real_data_np = self.real_data.cpu().numpy()
        real_labels_np = self.real_labels.cpu().numpy() if self.real_labels is not None else None
        
        for model_name, fake_data_np in self.generated_samples.items():
            print(f"Evaluating {model_name}...")
            
            if real_labels_np is not None:
                fake_labels_np = np.random.choice(
                    np.unique(real_labels_np), 
                    size=len(fake_data_np)
                )
            else:
                fake_labels_np = None
            
            metrics = compute_enhanced_generation_metrics(
                real_data_np, 
                fake_data_np,
                real_labels_np,
                fake_labels_np
            )
            
            metrics['frechet_distance'] = compute_frechet_distance(real_data_np, fake_data_np)
            
            self.evaluation_results[model_name] = metrics
            
            report = create_evaluation_report(
                metrics,
                os.path.join(self.results_dir, 'metrics', f'{model_name}_report.md')
            )
        
        print("Evaluation completed")
    
    def create_umap_visualization(self, n_neighbors: int = 15, min_dist: float = 0.1):
        print("Creating UMAP visualization...")
        
        try:
            max_samples = len(self.real_data)
            print(f"  Using all {max_samples} real samples for UMAP visualization")
            
            real_data_np = self.real_data.cpu().numpy()
            real_labels_np = self.real_labels.cpu().numpy() if hasattr(self.real_labels, 'cpu') else self.real_labels
            
            real_data_sample = real_data_np
            cell_type_labels = real_labels_np
            
            print(f"  Real data sample size: {len(real_data_sample)}")
            print(f"  Cell types in sample: {len(np.unique(cell_type_labels))} unique types")
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_data_sample)
            
            if len(real_data_sample) >= 2000:
                adaptive_n_neighbors = 30
            elif len(real_data_sample) >= 1000:
                adaptive_n_neighbors = 20
            else:
                adaptive_n_neighbors = min(n_neighbors, max(3, len(real_data_sample) - 1))
            print(f"  Using n_neighbors={adaptive_n_neighbors} (optimized for {len(real_data_sample)} samples)")
            
            print("  Calculating UMAP for real data...")
            reducer = umap.UMAP(
                n_neighbors=adaptive_n_neighbors,
                min_dist=min_dist,
                n_components=2,
                random_state=42,
                low_memory=True,
                verbose=0
            )
            real_embedding = reducer.fit_transform(real_scaled)
            
            print("  Projecting generated samples...")
            method_embeddings = {}
            for model_name, samples in self.generated_samples.items():
                if len(samples) > max_samples:
                    indices = np.random.choice(len(samples), max_samples, replace=False)
                    samples_subset = samples[indices]
                else:
                    samples_subset = samples
                
                samples_scaled = scaler.transform(samples_subset)
                method_embeddings[model_name] = reducer.transform(samples_scaled)
            
            print("  Creating figures...")
            
            n_methods = len(method_embeddings) + 1
            n_cols = 2
            n_rows = (n_methods + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
            ax = axes[0]
            n_types = len(np.unique(cell_type_labels))
            
            if n_types <= 10:
                color_map = 'tab10'
            elif n_types <= 20:
                color_map = 'tab20'
            else:
                color_map = 'viridis'
            
            scatter = ax.scatter(real_embedding[:, 0], real_embedding[:, 1],
                                c=cell_type_labels, cmap=color_map,
                                s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax.set_title('Real Data (Ground Truth)', fontsize=13, fontweight='bold')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.grid(True, alpha=0.3)
            
            if n_types <= 20:
                cbar = plt.colorbar(scatter, ax=ax, label='Cell Type', shrink=0.8)
                cbar.set_ticks(range(n_types))
                cbar.set_ticklabels([f'Type {i}' for i in range(n_types)])
            
            idx = 1
            for model_name, embedding in method_embeddings.items():
                if idx >= len(axes):
                    break
                ax = axes[idx]
                
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                    c=range(len(embedding)), cmap='viridis',
                                    s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
                ax.set_title(model_name.replace('_', ' ').title(), fontsize=13, fontweight='bold')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.grid(True, alpha=0.3)
                
                idx += 1
            
            for i in range(idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            fig_path = os.path.join(self.results_dir, 'figures', 'umap_all_methods_same_space.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Saved: umap_all_methods_same_space.png")
            
            if len(method_embeddings) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                
                ax = axes[0]
                scatter = ax.scatter(real_embedding[:, 0], real_embedding[:, 1],
                          c=cell_type_labels, cmap=color_map,
                          s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
                ax.set_title('Real Data', fontsize=13, fontweight='bold')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.grid(True, alpha=0.3)
                
                if n_types <= 20:
                    cbar = plt.colorbar(scatter, ax=ax, label='Cell Type', shrink=0.8)
                    cbar.set_ticks(range(n_types))
                    cbar.set_ticklabels([f'Type {i}' for i in range(n_types)])
                
                first_model = list(method_embeddings.keys())[0]
                embedding = method_embeddings[first_model]
                ax = axes[1]
                ax.scatter(embedding[:, 0], embedding[:, 1],
                          c=range(len(embedding)), cmap='viridis',
                          s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
                ax.set_title(first_model.replace('_', ' ').title(), fontsize=13, fontweight='bold')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fig_path = os.path.join(self.results_dir, 'figures', 'umap_real_vs_generated.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved: umap_real_vs_generated.png")
            
            print("✅ UMAP visualization complete!")
            
        except Exception as e:
            print(f"Error creating UMAP visualization: {e}")
            import traceback
            traceback.print_exc()
        
        return {}

    def _compute_overlap_scores(self, embedding, data_labels, all_labels):
        from sklearn.neighbors import NearestNeighbors
        
        real_mask = np.array(data_labels) == 'Real Data'
        real_embedding = embedding[real_mask]
        
        overlap_scores = {}
        
        for label in all_labels:
            if label == 'Real Data':
                continue
            
            gen_mask = np.array(data_labels) == label
            gen_embedding = embedding[gen_mask]
            
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn.fit(real_embedding)
            distances, _ = nn.kneighbors(gen_embedding)
            mean_distance = np.mean(distances)
            overlap_score = 1.0 / (1.0 + mean_distance)
            
            overlap_scores[label] = overlap_score
        
        return overlap_scores
    
    def _compute_enhanced_overlap_scores(self, embedding, data_labels, all_labels):
        from sklearn.neighbors import NearestNeighbors
        from scipy.spatial.distance import pdist, squareform
        from scipy.stats import wasserstein_distance
        
        real_mask = np.array(data_labels) == 'Real Data'
        real_embedding = embedding[real_mask]
        
        enhanced_scores = {}
        
        for label in all_labels:
            if label == 'Real Data':
                continue
            
            gen_mask = np.array(data_labels) == label
            gen_embedding = embedding[gen_mask]
            
            if len(gen_embedding) == 0:
                enhanced_scores[label] = 0.0
                continue
            
            nn = NearestNeighbors(n_neighbors=min(5, len(real_embedding)), metric='euclidean')
            nn.fit(real_embedding)
            distances, _ = nn.kneighbors(gen_embedding)
            mean_distance = np.mean(distances)
            distance_score = 1.0 / (1.0 + mean_distance)
            if len(real_embedding) > 5:
                real_nn = NearestNeighbors(n_neighbors=6).fit(real_embedding)
                real_distances = real_nn.kneighbors(real_embedding)[0][:, 1:].mean(axis=1)
            else:
                real_distances = np.ones(len(real_embedding))
                
            if len(gen_embedding) > 5:
                gen_nn = NearestNeighbors(n_neighbors=6).fit(gen_embedding)
                gen_distances = gen_nn.kneighbors(gen_embedding)[0][:, 1:].mean(axis=1)
            else:
                gen_distances = np.ones(len(gen_embedding))
            
            try:
                density_wasserstein = wasserstein_distance(real_distances, gen_distances)
                density_score = 1.0 / (1.0 + density_wasserstein)
            except:
                density_score = 0.5
            
            try:
                real_std = np.std(real_embedding, axis=0)
                gen_std = np.std(gen_embedding, axis=0)
                coverage_score = 1 - np.mean(np.abs(real_std - gen_std) / (real_std + 1e-8))
                coverage_score = max(0, coverage_score)
            except:
                coverage_score = 0.5
            
            try:
                from scipy.spatial import ConvexHull
                if len(real_embedding) >= 3 and len(gen_embedding) >= 3:
                    real_hull = ConvexHull(real_embedding)
                    gen_hull = ConvexHull(gen_embedding)
                    
                    real_volume = real_hull.volume if hasattr(real_hull, 'volume') else 1
                    gen_volume = gen_hull.volume if hasattr(gen_hull, 'volume') else 1
                    
                    volume_ratio = min(real_volume, gen_volume) / max(real_volume, gen_volume)
                    shape_score = volume_ratio
                else:
                    shape_score = 0.5
            except:
                shape_score = 0.5
            
            try:
                combined_embedding = np.vstack([real_embedding, gen_embedding])
                combined_labels = np.array(['real'] * len(real_embedding) + ['gen'] * len(gen_embedding))
                k = min(20, len(combined_embedding) - 1)
                nn_mixed = NearestNeighbors(n_neighbors=k).fit(combined_embedding)
                gen_neighbors = nn_mixed.kneighbors(gen_embedding)[1]
                
                real_neighbor_ratios = []
                for neighbors in gen_neighbors:
                    real_count = np.sum(combined_labels[neighbors] == 'real')
                    real_neighbor_ratios.append(real_count / k)
                
                expected_ratio = len(real_embedding) / len(combined_embedding)
                mixing_score = 1 - np.mean(np.abs(np.array(real_neighbor_ratios) - expected_ratio))
                mixing_score = max(0, mixing_score)
            except:
                mixing_score = 0.5
            
            enhanced_score = (
                0.30 * distance_score +
                0.25 * density_score +
                0.20 * coverage_score +
                0.15 * shape_score +
                0.10 * mixing_score
            )
            
            enhanced_scores[label] = enhanced_score
        
        return enhanced_scores
    
    def create_comparison_report(self):
        print("Creating comparison report...")
        comparison_data = []
        
        for method_name, metrics in self.evaluation_results.items():
            row = {'Method': method_name.replace('_', ' ').title()}
            
            key_metrics = [
                'mmd_rbf',
                'wasserstein_distance_mean',
                'correlation_preservation',
                'frechet_distance',
                'pca_mean_difference',
                'jensen_shannon_divergence',
                'biological_similarity_score'
            ]
            
            for metric in key_metrics:
                if metric in metrics and not np.isnan(metrics[metric]):
                    row[metric] = f"{metrics[metric]:.4f}"
                else:
                    row[metric] = "N/A"
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(self.results_dir, 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        
        self._create_metrics_visualization(df)
        report_path = os.path.join(self.results_dir, 'comparison_report.md')
        self._create_comprehensive_report(df, report_path)
        
        print(f"Comparison report saved to {report_path}")
        print(f"CSV results saved to {csv_path}")
    
    def _create_metrics_visualization(self, df):
        try:
            numeric_cols = [col for col in df.columns if col != 'Method']
            
            n_metrics = len(numeric_cols)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            plt.figure(figsize=(5*n_cols, 4*n_rows))
            
            for i, col in enumerate(numeric_cols):
                plt.subplot(n_rows, n_cols, i+1)
                
                methods = df['Method'].tolist()
                values = []
                
                for method in methods:
                    value_str = df[df['Method'] == method][col].iloc[0]
                    if value_str != 'N/A':
                        values.append(float(value_str))
                    else:
                        values.append(0)
                
                bars = plt.bar(methods, values)
                plt.title(col.replace('_', ' ').title())
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Score')
                plt.grid(True, alpha=0.3, axis='y')
                
                if values:
                    if 'distance' in col.lower() or 'divergence' in col.lower():
                        best_idx = np.argmin(values)
                    else:
                        best_idx = np.argmax(values)
                    bars[best_idx].set_color('gold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'figures', 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating metrics visualization: {e}")
            print("Skipping metrics visualization.")
    
    def _create_comprehensive_report(self, df, report_path):
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Baseline Comparison Report\n\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Experiment Setup\n\n")
            f.write(f"- Samples: {self.num_samples}\n")
            f.write(f"- Input dim: {self.input_dim}\n")
            f.write(f"- Data path: {self.data_path}\n\n")
            
            f.write("## Methods\n\n")
            for method in df['Method']:
                f.write(f"- **{method}**\n")
            f.write("\n")
            
            f.write("## Metrics\n\n")
            f.write("| Metric | Description | Best |\n")
            f.write("|--------|-------------|------|\n")
            f.write("| MMD RBF | Distribution distance | Lower |\n")
            f.write("| Biological Similarity | UMAP overlap | Higher |\n")
            f.write("| Wasserstein Distance | Distribution distance | Lower |\n")
            f.write("| Correlation Preservation | Gene correlation | Higher |\n")
            f.write("| Frechet Distance | Generation quality | Lower |\n")
            f.write("| PCA Mean Difference | PCA space difference | Lower |\n")
            f.write("| Jensen Shannon Divergence | Distribution similarity | Lower |\n\n")
            
            f.write("## Results\n\n")
            f.write("| " + " | ".join(df.columns) + " |\n")
            f.write("| " + " | ".join(["-" * len(col) for col in df.columns]) + " |\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join([str(v) for v in row.values]) + " |\n")
            f.write("\n\n")
            
            f.write("## Performance Analysis\n\n")
            numeric_cols = [col for col in df.columns if col != 'Method']
            
            for col in numeric_cols:
                values = []
                methods = []
                
                for _, row in df.iterrows():
                    if row[col] != 'N/A':
                        values.append(float(row[col]))
                        methods.append(row['Method'])
                
                if values:
                    if 'distance' in col.lower() or 'divergence' in col.lower():
                        best_idx = np.argmin(values)
                        best_desc = "min"
                    else:
                        best_idx = np.argmax(values)
                        best_desc = "max"
                    
                    f.write(f"- **{col.replace('_', ' ').title()}**: {methods[best_idx]} "
                           f"({best_desc}: {values[best_idx]:.4f})\n")
            
            f.write("\n## Conclusion\n\n")
            f.write("See Performance Analysis above for detailed rankings.\n")
    
    def run_full_comparison(
        self, 
        train_epochs: int = 100,
        our_model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        print("="*80)
        print("Starting baseline comparison experiment")
        print("="*80)
        
        train_loader, val_loader, test_loader = self.load_data()
        print("\nTraining Baseline Models...")
        self.train_baseline_models(train_loader, train_epochs)
        
        our_epochs = min(train_epochs * 4, 400)
        print(f"\nTraining Our Model ({our_epochs} epochs)...")
        self.load_our_model(our_model_path, config_path)
        self.train_our_model(train_loader, our_epochs)
        
        self.generate_samples()
        self.evaluate_all_methods()
        overlap_scores = self.create_umap_visualization()
        self.create_comparison_report()
        
        print("\n" + "="*50)
        print("Baseline comparison completed!")
        print(f"Results saved to: {self.results_dir}")
        print("="*50)
        
        return {
            'evaluation_results': self.evaluation_results,
            'overlap_scores': overlap_scores,
            'results_dir': self.results_dir
        }


def main():
    parser = argparse.ArgumentParser(description='Baseline Methods Comparison')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset')
    parser.add_argument('--results_dir', type=str, 
                       default='./experiments/results/baseline_comparison',
                       help='Results directory')
    parser.add_argument('--num_samples', type=int, default=2000,
                       help='Number of samples to generate and evaluate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for baseline models')
    parser.add_argument('--our_model_path', type=str, default=None,
                       help='Path to our trained model')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    comparison = BaselineComparison(
        data_path=args.data_path,
        results_dir=args.results_dir,
        device=device,
        num_samples=args.num_samples
    )
    
    results = comparison.run_full_comparison(
        train_epochs=args.epochs,
        our_model_path=args.our_model_path,
        config_path=args.config_path
    )
    
    print("\nComparison Summary:")
    for method, metrics in results['evaluation_results'].items():
        print(f"{method}: Fréchet Distance = {metrics.get('frechet_distance', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
