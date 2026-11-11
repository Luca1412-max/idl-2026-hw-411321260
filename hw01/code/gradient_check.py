# hw01/code/gradient_check.py
import numpy as np
from mlp import MLP

def compute_numerical_gradient(model, X, y, h=1e-5):
    """
    使用中央差分計算數值梯度：
    f'(theta) = (f(theta + h) - f(theta - h)) / (2 * h)
    """
    numerical_grads = {'weights': [], 'biases': []}
    
    # 針對每一層的 weights 和 biases 進行數值微擾
    for layer_idx in range(model.num_layers):
        # 複製當前權重與偏置的結構
        w_grad = np.zeros_like(model.weights[layer_idx])
        b_grad = np.zeros_like(model.biases[layer_idx])
        
        # 1. 檢查 weights 梯度
        W = model.weights[layer_idx]
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # f(theta + h)
                W[i, j] += h
                loss_plus = compute_loss(model, X, y)
                
                # f(theta - h)
                W[i, j] -= 2 * h
                loss_minus = compute_loss(model, X, y)
                
                # 復原權重
                W[i, j] += h
                
                w_grad[i, j] = (loss_plus - loss_minus) / (2 * h)
                
        # 2. 檢查 biases 梯度
        B = model.biases[layer_idx]
        for j in range(B.shape[1]):
            # f(theta + h)
            B[0, j] += h
            loss_plus = compute_loss(model, X, y)
            
            # f(theta - h)
            B[0, j] -= 2 * h
            loss_minus = compute_loss(model, X, y)
            
            # 復原偏置
            B[0, j] += h
            
            b_grad[0, j] = (loss_plus - loss_minus) / (2 * h)
            
        numerical_grads['weights'].append(w_grad)
        numerical_grads['biases'].append(b_grad)
        
    return numerical_grads

def compute_loss(model, X, y):
    """計算當前模型的整體 Loss（用於數值梯度比對）"""
    y_pred = model.forward(X)
    m = X.shape[0]
    if model.task == 'regression':
        # MSE Loss = 1/(2m) * sum((y_pred - y)^2)
        return 0.5 * np.mean((y_pred - y) ** 2)
    elif model.task == 'classification':
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # BCE Loss
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def gradient_check(task='classification'):
    print(f"=== 執行梯度檢查 (Task: {task}) ===")
    
    # 建立小規模網路進行檢查以節省計算時間
    input_dim = 2
    hidden_sizes = [4]
    output_dim = 1
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task=task, seed=42)
    
    # 產生少量測試數據
    np.random.seed(0)
    X = np.random.randn(5, input_dim)
    if task == 'regression':
        y = np.random.randn(5, output_dim)
    else:
        y = np.random.randint(0, 2, size=(5, output_dim)).astype(float)
        
    # 1. 計算解析梯度 (Backpropagation)
    analytic_grads = model.backward(X, y)
    
    # 2. 計算數值梯度 (Numerical Gradient)
    numerical_grads = compute_numerical_gradient(model, X, y, h=1e-5)
    
    # 3. 計算相對誤差 (Relative Error)
    max_relative_error = 0.0
    for l in range(model.num_layers):
        # Weight error
        w_an = analytic_grads['weights'][l]
        w_num = numerical_grads['weights'][l]
        numerator = np.linalg.norm(w_an - w_num)
        denominator = np.linalg.norm(w_an) + np.linalg.norm(w_num)
        diff_w = numerator / (denominator + 1e-12)
        
        # Bias error
        b_an = analytic_grads['biases'][l]
        b_num = numerical_grads['biases'][l]
        numerator_b = np.linalg.norm(b_an - b_num)
        denominator_b = np.linalg.norm(b_an) + np.linalg.norm(b_num)
        diff_b = numerator_b / (denominator_b + 1e-12)
        
        layer_max = max(diff_w, diff_b)
        max_relative_error = max(max_relative_error, layer_max)
        print(f"Layer {l+1} 最大相對誤差: W = {diff_w:.2e}, b = {diff_b:.2e}")
        
    print(f"整體最高相對誤差: {max_relative_error:.2e}")
    if max_relative_error < 1e-5:
        print(" 梯度檢查通過！解析梯度與數值梯度高度一致。")
    else:
        print(" 梯度誤差偏高，請檢查反向傳播公式！")

if __name__ == "__main__":
    gradient_check('classification')
    print("-" * 40)
    gradient_check('regression')