# hw01/code/gradient_check.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP

def compute_numerical_gradient(model, X, y, h=1e-5):
    """
    使用中央差分計算數值梯度：
    f'(theta) = (f(theta + h) - f(theta - h)) / (2 * h)
    """
    numerical_grads = {'weights': [], 'biases': []}
    
    for layer_idx in range(model.num_layers):
        w_grad = np.zeros_like(model.weights[layer_idx])
        b_grad = np.zeros_like(model.biases[layer_idx])
        
        # 1. 檢查 weights 梯度
        W = model.weights[layer_idx]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += h
                loss_plus = compute_loss(model, X, y)
                W[i, j] -= 2 * h
                loss_minus = compute_loss(model, X, y)
                W[i, j] += h
                w_grad[i, j] = (loss_plus - loss_minus) / (2 * h)
                
        # 2. 檢查 biases 梯度
        B = model.biases[layer_idx]
        for j in range(B.shape[1]):
            B[0, j] += h
            loss_plus = compute_loss(model, X, y)
            B[0, j] -= 2 * h
            loss_minus = compute_loss(model, X, y)
            B[0, j] += h
            b_grad[0, j] = (loss_plus - loss_minus) / (2 * h)
            
        numerical_grads['weights'].append(w_grad)
        numerical_grads['biases'].append(b_grad)
        
    return numerical_grads

def compute_loss(model, X, y):
    y_pred = model.forward(X)
    if model.task == 'regression':
        return 0.5 * np.mean((y_pred - y) ** 2)
    elif model.task == 'classification':
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def gradient_check(task='classification'):
    print(f"=== 執行梯度檢查 (Task: {task}) ===")
    
    input_dim = 2
    hidden_sizes = [4]
    output_dim = 1
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task=task, seed=42)
    
    np.random.seed(0)
    X = np.random.randn(5, input_dim)
    if task == 'regression':
        y = np.random.randn(5, output_dim)
    else:
        y = np.random.randint(0, 2, size=(5, output_dim)).astype(float)
        
    analytic_grads = model.backward(X, y)
    numerical_grads = compute_numerical_gradient(model, X, y, h=1e-5)
    
    layer_errors = []
    max_relative_error = 0.0
    for l in range(model.num_layers):
        w_an = analytic_grads['weights'][l]
        w_num = numerical_grads['weights'][l]
        diff_w = np.linalg.norm(w_an - w_num) / (np.linalg.norm(w_an) + np.linalg.norm(w_num) + 1e-12)
        
        b_an = analytic_grads['biases'][l]
        b_num = numerical_grads['biases'][l]
        diff_b = np.linalg.norm(b_an - b_num) / (np.linalg.norm(b_an) + np.linalg.norm(b_num) + 1e-12)
        
        layer_max = max(diff_w, diff_b)
        layer_errors.append(layer_max)
        max_relative_error = max(max_relative_error, layer_max)
        print(f"Layer {l+1} 最大相對誤差: W = {diff_w:.2e}, b = {diff_b:.2e}")
        
    print(f"整體最高相對誤差: {max_relative_error:.2e}")
    return layer_errors

def plot_gradient_check_results():
    tasks = ['classification', 'regression']
    errors_dict = {}
    for task in tasks:
        print(f"\nRunning check for {task}...")
        input_dim = 2
        hidden_sizes = [4]
        output_dim = 1
        model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task=task, seed=42)
        np.random.seed(0)
        X = np.random.randn(5, input_dim)
        y = np.random.randn(5, output_dim) if task == 'regression' else np.random.randint(0, 2, size=(5, output_dim)).astype(float)
        
        analytic_grads = model.backward(X, y)
        numerical_grads = compute_numerical_gradient(model, X, y, h=1e-5)
        
        layer_max_errs = []
        for l in range(model.num_layers):
            w_an, w_num = analytic_grads['weights'][l], numerical_grads['weights'][l]
            b_an, b_num = analytic_grads['biases'][l], numerical_grads['biases'][l]
            err_w = np.linalg.norm(w_an - w_num) / (np.linalg.norm(w_an) + np.linalg.norm(w_num) + 1e-12)
            err_b = np.linalg.norm(b_an - b_num) / (np.linalg.norm(b_an) + np.linalg.norm(b_num) + 1e-12)
            layer_max_errs.append(max(err_w, err_b))
        errors_dict[task] = layer_max_errs

    # 繪製柱狀圖
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = [f'Layer {i+1}' for i in range(len(errors_dict['classification']))]
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, errors_dict['classification'], width, label='Classification (BCE)', color='skyblue')
    ax.bar(x + width/2, errors_dict['regression'], width, label='Regression (MSE)', color='salmon')
    
    ax.set_ylabel('Max Relative Error')
    ax.set_title('Gradient Checking: Analytic vs Numerical Gradients')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_yscale('log')
    ax.axhline(y=1e-5, color='r', linestyle='--', label='Threshold (1e-5)')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    os.makedirs('../figures', exist_ok=True)

    plt.tight_layout()
    plt.savefig('../figures/gradient_check.png', dpi=300)
    print("\n✅ 梯度檢查圖表已成功儲存至 hw01/figures/gradient_check.png")

if __name__ == "__main__":
    plot_gradient_check_results()