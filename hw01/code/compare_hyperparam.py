# hw01/code/compare_hyperparam.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP
from generate_data import generate_spiral_data

def compute_accuracy(y_pred, y_true):
    preds = (y_pred >= 0.5).astype(int)
    return np.mean(preds == y_true.astype(int))

def run_hyperparameter_experiment():
    # 產生嚴格 noise=0.035 的雙螺旋資料
    X_tr, y_tr = generate_spiral_data(1000, 0.035, 42)
    X_va, y_va = generate_spiral_data(250, 0.035, 100)
    
    # 依照助教建議，拉高 epoch 與測試多組 lr
    epochs = 1500
    lrs = [0.1, 0.5, 1.0]
    
    results = {}
    eps = 1e-15
    
    for lr in lrs:
        np.random.seed(42)
        model = MLP(input_dim=2, hidden_sizes=[32, 32], output_dim=1, task='classification', seed=42)
        val_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            indices = np.arange(X_tr.shape[0])
            np.random.shuffle(indices)
            for start in range(0, X_tr.shape[0], 64):
                end = min(start + 64, X_tr.shape[0])
                grads = model.backward(X_tr[indices[start:end]], y_tr[indices[start:end]])
                model.update(grads, lr)
                
            p_va = np.clip(model.forward(X_va), eps, 1-eps)
            loss_va = -np.mean(y_va * np.log(p_va) + (1 - y_va) * np.log(1 - p_va))
            acc_va = compute_accuracy(p_va, y_va)
            
            val_losses.append(loss_va)
            val_accs.append(acc_va)
            
        results[lr] = (val_losses, val_accs)
        print(f"LR = {lr} 訓練完成！最終驗證 Accuracy: {val_accs[-1]:.4f}")
        
    # 繪製並更新 hyperparam_comparison.png
    os.makedirs('../figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for lr in lrs:
        losses, accs = results[lr]
        axes[0].plot(losses, label=f'LR = {lr}')
        axes[1].plot(accs, label=f'LR = {lr}')
        
    axes[0].set_title('Validation Loss Comparison (Noise=0.035)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)
    
    axes[1].set_title('Validation Accuracy Comparison (Noise=0.035)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../figures/hyperparam_comparison.png', dpi=300)
    plt.close()
    print("✅ hyperparam_comparison.png 已成功使用新超參數重畫！")

if __name__ == "__main__":
    run_hyperparameter_experiment()