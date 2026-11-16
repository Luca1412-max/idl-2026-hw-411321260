# hw01/code/train_regression.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP
from generate_data import generate_regression_data, generate_spiral_data

def train_regression_models():
    X_reg, y_reg = generate_regression_data(3000, seed=42)
    X_train, y_train = X_reg[:2000], y_reg[:2000]
    X_val, y_val = X_reg[2000:2500], y_reg[2000:2500]
    X_test, y_test = X_reg[2500:], y_reg[2500:]
    
    epochs = 1000
    lr = 0.2
    batch_size = 64
    num_samples = X_train.shape[0]
    
    # 2-Layer Regression
    model_2l = MLP(input_dim=2, hidden_sizes=[32], output_dim=1, task='regression', seed=42)
    train_l_2, val_l_2 = [], []
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            grads = model_2l.backward(X_train[indices[start:end]], y_train[indices[start:end]])
            model_2l.update(grads, lr)
        train_l_2.append(0.5 * np.mean((model_2l.forward(X_train) - y_train)**2))
        val_l_2.append(0.5 * np.mean((model_2l.forward(X_val) - y_val)**2))
        
    # 3-Layer Regression
    model_3l = MLP(input_dim=2, hidden_sizes=[32, 32], output_dim=1, task='regression', seed=42)
    train_l_3, val_l_3 = [], []
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            grads = model_3l.backward(X_train[indices[start:end]], y_train[indices[start:end]])
            model_3l.update(grads, lr)
        train_l_3.append(0.5 * np.mean((model_3l.forward(X_train) - y_train)**2))
        val_l_3.append(0.5 * np.mean((model_3l.forward(X_val) - y_val)**2))
        
    return model_2l, model_3l, (train_l_2, val_l_2), (train_l_3, val_l_3), (X_test, y_test)

def train_spiral_loss_for_curves():
    X_tr, y_tr = generate_spiral_data(1000, 0.035, 42)
    X_va, y_va = generate_spiral_data(250, 0.035, 100)
    
    eps = 1e-15
    epochs = 1500
    lr = 0.5
    
    # 2-Layer Spiral Loss
    model_2l = MLP(input_dim=2, hidden_sizes=[32], output_dim=1, task='classification', seed=42)
    train_l_2, val_l_2 = [], []
    for epoch in range(epochs):
        indices = np.arange(X_tr.shape[0])
        np.random.shuffle(indices)
        for start in range(0, X_tr.shape[0], 64):
            end = min(start + 64, X_tr.shape[0])
            grads = model_2l.backward(X_tr[indices[start:end]], y_tr[indices[start:end]])
            model_2l.update(grads, lr)
        p_tr = np.clip(model_2l.forward(X_tr), eps, 1-eps)
        p_va = np.clip(model_2l.forward(X_va), eps, 1-eps)
        train_l_2.append(-np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr)))
        # 修正：將原本誤寫的 y_va 改為 p_va，解決 nan 問題
        val_l_2.append(-np.mean(y_va * np.log(p_va) + (1 - y_va) * np.log(1 - p_va)))
        
    # 3-Layer Spiral Loss
    model_3l = MLP(input_dim=2, hidden_sizes=[32, 32], output_dim=1, task='classification', seed=42)
    train_l_3, val_l_3 = [], []
    for epoch in range(epochs):
        indices = np.arange(X_tr.shape[0])
        np.random.shuffle(indices)
        for start in range(0, X_tr.shape[0], 64):
            end = min(start + 64, X_tr.shape[0])
            grads = model_3l.backward(X_tr[indices[start:end]], y_tr[indices[start:end]])
            model_3l.update(grads, lr)
        p_tr = np.clip(model_3l.forward(X_tr), eps, 1-eps)
        p_va = np.clip(model_3l.forward(X_va), eps, 1-eps)
        train_l_3.append(-np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr)))
        val_l_3.append(-np.mean(y_va * np.log(p_va) + (1 - y_va) * np.log(1 - p_va)))
        
    return (train_l_2, val_l_2), (train_l_3, val_l_3)

def plot_combined_loss_curves(reg_curves_2, reg_curves_3, spiral_curves_2, spiral_curves_3):
    os.makedirs('../figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(reg_curves_2[0], linestyle='--', color='blue', label='2-Layer Train (MSE)')
    axes[0].plot(reg_curves_2[1], linestyle='-', color='blue', label='2-Layer Val (MSE)')
    axes[0].plot(reg_curves_3[0], linestyle='--', color='orange', label='3-Layer Train (MSE)')
    axes[0].plot(reg_curves_3[1], linestyle='-', color='orange', label='3-Layer Val (MSE)')
    axes[0].set_title('Regression Loss Curves (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss (Log Scale)')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    
    axes[1].plot(spiral_curves_2[0], linestyle='--', color='blue', label='2-Layer Train (BCE)')
    axes[1].plot(spiral_curves_2[1], linestyle='-', color='blue', label='2-Layer Val (BCE)')
    axes[1].plot(spiral_curves_3[0], linestyle='--', color='orange', label='3-Layer Train (BCE)')
    axes[1].plot(spiral_curves_3[1], linestyle='-', color='orange', label='3-Layer Val (BCE)')
    axes[1].set_title('Spiral Classification Loss Curves (BCE, noise=0.035)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BCE Loss')
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('../figures/loss_curves.png', dpi=300)
    plt.close()
    print("✅ 整合後的 loss_curves.png 已更新成功！")

def plot_regression_comparison(model_2l, model_3l, test_data):
    # 直接用數學函數計算真實曲面，避免插值尖峰
    xx, yy = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    x1, x2 = xx, yy
    
    term1 = 0.45 * np.sin(2.5 * np.pi * x1) * np.cos(1.5 * np.pi * x2)
    term2 = 0.25 * np.sin(3.0 * np.pi * x1 * x2)
    term3 = 0.20 * np.exp(-8.0 * ((x1 - 0.35)**2 + (x2 + 0.25)**2))
    term4 = -0.20 * np.exp(-10.0 * ((x1 + 0.40)**2 + (x2 - 0.35)**2))
    term5 = 0.15 * (x1**2 - x2**2)
    Z_true = term1 + term2 + term3 + term4 + term5
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_2l = model_2l.forward(grid_points).reshape(xx.shape)
    Z_3l = model_3l.forward(grid_points).reshape(xx.shape)
    
    vmin = min(Z_true.min(), Z_2l.min(), Z_3l.min())
    vmax = max(Z_true.max(), Z_2l.max(), Z_3l.max())
    
    err_2l = np.abs(Z_2l - Z_true)
    err_3l = np.abs(Z_3l - Z_true)
    err_vmin = min(err_2l.min(), err_3l.min())
    err_vmax = max(err_2l.max(), err_3l.max())
    
    fig = plt.figure(figsize=(25, 5))
    
    ax1 = fig.add_subplot(1, 5, 1, projection='3d')
    ax1.plot_surface(xx, yy, Z_true, cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
    ax1.set_title('True Surface')
    ax1.set_zlim(vmin, vmax)
    
    ax2 = fig.add_subplot(1, 5, 2, projection='3d')
    ax2.plot_surface(xx, yy, Z_2l, cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
    ax2.set_title('2-Layer Prediction')
    ax2.set_zlim(vmin, vmax)
    
    ax3 = fig.add_subplot(1, 5, 3, projection='3d')
    ax3.plot_surface(xx, yy, Z_3l, cmap='viridis', vmin=vmin, vmax=vmax, edgecolor='none', alpha=0.9)
    ax3.set_title('3-Layer Prediction')
    ax3.set_zlim(vmin, vmax)
    
    ax4 = fig.add_subplot(1, 5, 4, projection='3d')
    ax4.plot_surface(xx, yy, err_2l, cmap='plasma', vmin=err_vmin, vmax=err_vmax, edgecolor='none', alpha=0.9)
    ax4.set_title('2-Layer Absolute Error')
    
    ax5 = fig.add_subplot(1, 5, 5, projection='3d')
    ax5.plot_surface(xx, yy, err_3l, cmap='plasma', vmin=err_vmin, vmax=err_vmax, edgecolor='none', alpha=0.9)
    ax5.set_title('3-Layer Absolute Error')
    
    plt.tight_layout()
    plt.savefig('../figures/regression_comparison.png', dpi=300)
    plt.close()
    print("✅ regression_comparison.png 已更新成功（含兩層與三層誤差圖）！")

if __name__ == "__main__":
    print("正在訓練迴歸模型...")
    model_2l, model_3l, reg_2, reg_3, test_data = train_regression_models()
    
    print("正在訓練雙螺旋分類模型以取得 Loss 曲線...")
    spiral_2, spiral_3 = train_spiral_loss_for_curves()
    
    print("正在繪製結合後的 loss_curves.png...")
    plot_combined_loss_curves(reg_2, reg_3, spiral_2, spiral_3)
    
    print("正在繪製 regression_comparison.png...")
    plot_regression_comparison(model_2l, model_3l, test_data)
    
    test_mse_2l = 0.5 * np.mean((model_2l.forward(test_data[0]) - test_data[1])**2)
    test_mse_3l = 0.5 * np.mean((model_3l.forward(test_data[0]) - test_data[1])**2)
    print(f"\n[實際輸出結果 - 迴歸任務]")
    print(f"2-Layer MLP Test MSE: {test_mse_2l:.4f}")
    print(f"3-Layer MLP Test MSE: {test_mse_3l:.4f}")