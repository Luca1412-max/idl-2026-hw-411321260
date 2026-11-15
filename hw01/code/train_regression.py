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
    
    # 2-Layer Regression
    model_2l = MLP(input_dim=2, hidden_sizes=[32], output_dim=1, task='regression', seed=42)
    train_l_2, val_l_2 = [], []
    epochs = 300
    lr = 0.05
    batch_size = 64
    num_samples = X_train.shape[0]
    
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
    model = MLP(input_dim=2, hidden_sizes=[32, 32], output_dim=1, task='classification', seed=42)
    train_losses, val_losses = [], []
    eps = 1e-15
    for epoch in range(300):
        indices = np.arange(X_tr.shape[0])
        np.random.shuffle(indices)
        for start in range(0, X_tr.shape[0], 64):
            end = min(start + 64, X_tr.shape[0])
            grads = model.backward(X_tr[indices[start:end]], y_tr[indices[start:end]])
            model.update(grads, 0.1)
        p_tr = np.clip(model.forward(X_tr), eps, 1-eps)
        p_va = np.clip(model.forward(X_va), eps, 1-eps)
        train_losses.append(-np.mean(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr)))
        val_losses.append(-np.mean(y_va * np.log(p_va) + (1 - y_va) * np.log(1 - p_va)))
    return train_losses, val_losses

def plot_combined_loss_curves(reg_curves_2, reg_curves_3, spiral_curves):
    os.makedirs('../figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：迴歸任務 MSE Loss
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
    
    # 右圖：雙螺旋分類任務 BCE Loss
    axes[1].plot(spiral_curves[0], linestyle='--', color='green', label='3-Layer Train (BCE)')
    axes[1].plot(spiral_curves[1], linestyle='-', color='green', label='3-Layer Val (BCE)')
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
    X_test, y_test = test_data
    y_pred_2l = model_2l.forward(X_test).flatten()
    y_pred_3l = model_3l.forward(X_test).flatten()
    y_true = y_test.flatten()
    
    fig = plt.figure(figsize=(16, 4))
    
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(X_test[:, 0], X_test[:, 1], y_true, c=y_true, cmap='viridis', s=15)
    ax1.set_title('True Surface')
    
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(X_test[:, 0], X_test[:, 1], y_pred_2l, c=y_pred_2l, cmap='viridis', s=15)
    ax2.set_title('2-Layer Prediction')
    
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(X_test[:, 0], X_test[:, 1], y_pred_3l, c=y_pred_3l, cmap='viridis', s=15)
    ax3.set_title('3-Layer Prediction')
    
    ax4 = fig.add_subplot(1, 4, 4)
    err_2l = np.abs(y_pred_2l - y_true)
    err_3l = np.abs(y_pred_3l - y_true)
    ax4.hist(err_2l, bins=30, alpha=0.5, color='blue', label='2-Layer Error')
    ax4.hist(err_3l, bins=30, alpha=0.5, color='orange', label='3-Layer Error')
    ax4.set_title('Absolute Error Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/regression_comparison.png', dpi=300)
    plt.close()
    print("✅ regression_comparison.png 已更新成功！")

if __name__ == "__main__":
    print("正在訓練迴歸模型...")
    model_2l, model_3l, reg_2, reg_3, test_data = train_regression_models()
    
    print("正在訓練雙螺旋分類模型以取得 Loss 曲線...")
    spiral_train_l, spiral_val_l = train_spiral_loss_for_curves()
    
    print("正在繪製結合後的 loss_curves.png...")
    plot_combined_loss_curves(reg_2, reg_3, (spiral_train_l, spiral_val_l))
    
    print("正在繪製 regression_comparison.png...")
    plot_regression_comparison(model_2l, model_3l, test_data)
    
    test_mse_2l = 0.5 * np.mean((model_2l.forward(test_data[0]) - test_data[1])**2)
    test_mse_3l = 0.5 * np.mean((model_3l.forward(test_data[0]) - test_data[1])**2)
    print(f"\n[實際輸出結果 - 迴歸任務]")
    print(f"2-Layer MLP Test MSE: {test_mse_2l:.4f}")
    print(f"3-Layer MLP Test MSE: {test_mse_3l:.4f}")