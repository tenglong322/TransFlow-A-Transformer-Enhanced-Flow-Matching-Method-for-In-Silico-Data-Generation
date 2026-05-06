import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
import scanpy as sc
import sys, warnings
warnings.filterwarnings('ignore')

# ========== Metric Functions ==========
def mmd_rbf(X, Y, n_sub=500):
    n = min(n_sub, len(X), len(Y))
    Xs = X[np.random.choice(len(X), n, replace=False)]
    Ys = Y[np.random.choice(len(Y), n, replace=False)]
    d = cdist(Xs[:200], Ys[:200], 'sqeuclidean')
    gamma = 1.0 / max(np.median(d), 1e-6)
    XX = rbf_kernel(Xs, Xs, gamma)
    YY = rbf_kernel(Ys, Ys, gamma)
    XY = rbf_kernel(Xs, Ys, gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())

def wass_avg(X, Y, nf=100):
    idx = np.random.choice(X.shape[1], min(nf, X.shape[1]), replace=False)
    return float(np.mean([wasserstein_distance(X[:,i], Y[:,i]) for i in idx]))

def js_div(X, Y, nb=50, nf=100):
    idx = np.random.choice(X.shape[1], min(nf, X.shape[1]), replace=False)
    js = []
    for i in idx:
        lo, hi = min(X[:,i].min(), Y[:,i].min()), max(X[:,i].max(), Y[:,i].max())
        if hi == lo: js.append(0); continue
        bins = np.linspace(lo, hi, nb+1)
        p, _ = np.histogram(X[:,i], bins=bins, density=True)
        q, _ = np.histogram(Y[:,i], bins=bins, density=True)
        p, q = p/(p.sum()+1e-10), q/(q.sum()+1e-10)
        m = 0.5*(p+q)
        v = 0
        for pi,qi,mi in zip(p,q,m):
            if mi>0:
                if pi>0: v += 0.5*pi*np.log(pi/mi+1e-10)
                if qi>0: v += 0.5*qi*np.log(qi/mi+1e-10)
        js.append(max(0,v))
    return float(np.mean(js))

def knn_acc(X, Y, k=5, n_sub=500):
    n = min(n_sub, len(X), len(Y))
    Xs = X[np.random.choice(len(X), n, replace=False)]
    Ys = Y[np.random.choice(len(Y), n, replace=False)]
    c = np.vstack([Xs, Ys])
    l = np.array([0]*n + [1]*n)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(c, l)
    return float(knn.score(c, l))

# ========== Table rendering ==========
def make_table(results, morder, dname, path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    ax.set_title(f'{dname} - Comprehensive Metrics', fontsize=14, fontweight='bold', pad=20)
    cols = ['Method', 'MMD ↓', 'Wass ↓', 'JS Div ↓', 'Corr(Mean) ↑', 'Corr(Var) ↑', 'Sparsity ↑', 'KNN →0.5']
    keys = ['MMD','Wass','JS','CorrM','CorrV','Spar','KNN']
    
    best = {}
    for k in keys:
        vals = [results[m][k] for m in morder]
        if k in ['MMD','Wass','JS']: best[k] = min(vals)
        elif k == 'KNN': best[k] = min(vals, key=lambda x: abs(x-0.5))
        else: best[k] = max(vals)
    
    ct, cc = [], []
    for m in morder:
        r = results[m]
        row = [m]
        rc = ['#FFFFFF']*8
        if m == 'TransFlow': rc[0] = '#FFE4B5'
        for j,k in enumerate(keys):
            v = r[k]; row.append(f'{v:.4f}')
            ib = False
            if k=='KNN': ib = abs(v-0.5) <= abs(best[k]-0.5)+1e-6
            elif k in ['MMD','Wass','JS']: ib = v <= best[k]+1e-6
            else: ib = v >= best[k]-1e-6
            if ib: rc[j+1] = '#90EE90'
        ct.append(row); cc.append(rc)
    
    t = ax.table(cellText=ct, colLabels=cols, cellColours=cc,
                 colColours=['#4472C4']*8, cellLoc='center', loc='center')
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 1.8)
    for key, cell in t.get_celld().items():
        i, j = key
        if i == 0: cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#D0D0D0')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved: {path}'); plt.close()

# ========== Main ==========
if __name__ == '__main__':
    dataset = sys.argv[1]
    data_dir = f'/root/metrics_work/{dataset}'
    np.random.seed(42)
    
    mfiles = {
        'TransFlow': 'transflow_samples.npy',
        'scVI': 'scvi_samples.npy',
        'scGAN': 'scgan_samples.npy',
        'cellFLOW': 'cellflow_samples.npy',
        'scDiffusion': 'scdiffusion_samples.npy',
    }
    
    if dataset.startswith('ad'):
        adata = sc.read_h5ad('/root/data/human_ad/human_ad_7types.h5ad')
        vmean = np.array(adata.var['mean'])
        vstd = np.array(adata.var['std']); vstd[vstd==0] = 1.0
        
        real_sc = np.load(f'{data_dir}/human_ad_X.npy')  # scaled
        real_un = real_sc * vstd + vmean                   # unscaled
        
        methods_sc, methods_un = {}, {}
        for m, f in mfiles.items():
            gs = np.load(f'{data_dir}/{f}')
            methods_sc[m] = gs
            methods_un[m] = gs * vstd + vmean
        dname = 'Human AD (v33)' if 'v33' in dataset else 'Human AD (v6)'
        
    elif dataset == 'pbmc':
        adata = sc.read_h5ad(f'{data_dir}/pbmc_real.h5ad')
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=10000)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        
        # Unscaled real data (normalize+log1p, has zeros)
        if hasattr(adata.X, 'toarray'):
            real_un = adata.X.toarray().astype(np.float32)
        else:
            real_un = adata.X.astype(np.float32)
        
        # Scale and get params
        sc.pp.scale(adata)
        if hasattr(adata.X, 'toarray'):
            real_sc = adata.X.toarray().astype(np.float32)
        else:
            real_sc = adata.X.astype(np.float32)
        
        vmean = np.array(adata.var['mean'])
        vstd = np.array(adata.var['std']); vstd[vstd==0] = 1.0
        
        methods_sc, methods_un = {}, {}
        for m, f in mfiles.items():
            gs = np.load(f'{data_dir}/{f}')
            methods_sc[m] = gs
            methods_un[m] = gs * vstd + vmean  # reverse scale
        dname = 'PBMC Data'
    
    print(f'=== {dname} ===')
    print(f'Real scaled: {real_sc.shape}, range [{real_sc.min():.2f},{real_sc.max():.2f}]')
    print(f'Real unscaled: {real_un.shape}, range [{real_un.min():.2f},{real_un.max():.2f}], zeros: {(real_un==0).mean():.4f}')
    
    morder = ['TransFlow', 'scVI', 'scGAN', 'cellFLOW', 'scDiffusion']
    all_res = {}
    for m in morder:
        gs = methods_sc[m]
        gu = methods_un[m]
        n = min(len(real_sc), len(gs))
        rs, ru = real_sc[:n].astype(np.float64), real_un[:n].astype(np.float64)
        gs64, gu64 = gs[:n].astype(np.float64), gu[:n].astype(np.float64)
        
        # Distribution metrics on SCALED data
        mmd = mmd_rbf(rs, gs64)
        ws = wass_avg(rs, gs64)
        js = js_div(rs, gs64)
        knn = knn_acc(rs, gs64)
        # Correlation + Sparsity on UNSCALED data
        cm = float(np.corrcoef(ru.mean(0), gu64.mean(0))[0,1])
        cv = float(np.corrcoef(ru.var(0), gu64.var(0))[0,1])
        sp = float(1 - abs((ru==0).mean() - (gu64<=0).mean()))
        
        all_res[m] = {'MMD':mmd, 'Wass':ws, 'JS':js, 'CorrM':cm, 'CorrV':cv, 'Spar':sp, 'KNN':knn}
        print(f'{m}: MMD={mmd:.4f} Wass={ws:.4f} JS={js:.4f} CorrM={cm:.4f} CorrV={cv:.4f} Spar={sp:.4f} KNN={knn:.4f}')
    
    make_table(all_res, morder, dname, f'/root/metrics_work/{dataset}_metrics_table.png')
    print('Done')
