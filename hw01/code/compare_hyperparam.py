# hw01/code/compare_hyperparam.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP
from train_spiral import prepare_spiral_datasets, compute_bce_loss, compute_accuracy

def run_hyperparam_experiment():
    train_data, val_data, _ = prepare_spiral_datasets()
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # 測試兩種不同的學習率設定
    learning_rates = [0.05, 0.2]
    history = {}
    
    for lr in learning_rates:
        print(f"正在訓練三層 MLP (LR = {lr})...")
        model = MLP(input_dim=2, hidden_sizes=[32, 32], output_dim=1, task='classification', seed=42)
        
        val_accs = []
        val_losses = []
        epochs = 300
        batch_size = 64
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                grads = model.backward(X_shuffled[start:end], y_shuffled[start:end])
                model.update(grads, lr)
                
            val_pred = model.forward(X_val)
            val_losses.append(compute_bce_loss(val_pred, y_val))
            val_accs.append(compute_accuracy(val_pred, y_val))
            
        history[lr] = {'loss': val_losses, 'acc': val_accs}

    # 繪製對比圖
    os.makedirs('../figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for lr in learning_rates:
        axes[0].plot(history[lr]['loss'], label=f'LR = {lr}')
        axes[1].plot(history[lr]['acc'], label=f'LR = {lr}')
        
    axes[0].set_title('Validation Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)
    
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../figures/hyperparam_comparison.png', dpi=300)
    plt.close()
    print("✅ 超參數對比圖已成功儲存至 hw01/figures/hyperparam_comparison.png")

if __name__ == "__main__":
    run_hyperparam_experiment()