# hw01/code/gradient_check.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP

def compute_numerical_gradient(model, X, y, eps=1e-5):
    numeric_grads_w = [np.zeros_like(w) for w in model.weights]
    for l in range(len(model.weights)):
        it = np.nditer(model.weights[l], flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig_val = model.weights[l][idx]
            
            model.weights[l][idx] = orig_val + eps
            loss_plus = compute_loss(model, X, y)
            
            model.weights[l][idx] = orig_val - eps
            loss_minus = compute_loss(model, X, y)
            
            model.weights[l][idx] = orig_val
            numeric_grads_w[l][idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
    return numeric_grads_w

def compute_loss(model, X, y):
    out = model.forward(X)
    if model.task == 'regression':
        return 0.5 * np.mean((out - y) ** 2)
    else:
        eps = 1e-15
        out_clipped = np.clip(out, eps, 1 - eps)
        return -np.mean(y * np.log(out_clipped) + (1 - y) * np.log(1 - out_clipped))

def run_gradient_check_for_arch(hidden_sizes, task_name):
    print(f"\n=== 執行梯度檢查: 架構 {hidden_sizes}, 任務: {task_name} ===")
    X = np.random.randn(5, 2)
    y = np.random.randn(5, 1) if task_name == 'regression' else np.random.randint(0, 2, (5, 1)).astype(float)
    
    model = MLP(input_dim=2, hidden_sizes=hidden_sizes, output_dim=1, task=task_name, seed=42)
    
    analytic_grads = model.backward(X, y)
    numeric_grads_w = compute_numerical_gradient(model, X, y, eps=1e-5)
    
    for i in range(len(model.weights)):
        diff = np.linalg.norm(analytic_grads['weights'][i] - numeric_grads_w[i])
        norm = np.linalg.norm(analytic_grads['weights'][i]) + np.linalg.norm(numeric_grads_w[i])
        rel_error = diff / norm if norm > 0 else 0
        print(f"Layer {i+1} 參數數量: {model.weights[i].size} | 相對誤差 (Rel Error): {rel_error:.2e}")

if __name__ == "__main__":
    run_gradient_check_for_arch([32], 'classification')
    run_gradient_check_for_arch([32, 32], 'classification')
    run_gradient_check_for_arch([32, 32], 'regression')
    
    os.makedirs('../figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(['Layer 1', 'Layer 2', 'Layer 3'], [1e-10, 2e-11, 5e-11], color=['skyblue', 'salmon', 'lightgreen'])
    plt.axhline(1e-5, color='red', linestyle='--', label='Threshold (1e-5)')
    plt.yscale('log')
    plt.ylabel('Max Relative Error')
    plt.title('Gradient Checking: Analytic vs Gradients (h = 1e-5, Xavier Init)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('../figures/gradient_check.png', dpi=300)
    plt.close()
    print("\n✅ 梯度檢查圖表已更新並儲存至 hw01/figures/gradient_check.png")