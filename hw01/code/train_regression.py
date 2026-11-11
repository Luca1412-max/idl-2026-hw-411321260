# hw01/code/train_regression.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP

def generate_data(num_samples=3000, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(num_samples, 2))
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    
    term1 = 0.45 * np.sin(2.5 * np.pi * x1) * np.cos(1.5 * np.pi * x2)
    term2 = 0.25 * np.sin(3.0 * np.pi * x1 * x2)
    term3 = 0.20 * np.exp(-8.0 * ((x1 - 0.35)**2 + (x2 + 0.25)**2))
    term4 = -0.20 * np.exp(-10.0 * ((x1 + 0.40)**2 + (x2 - 0.35)**2))
    term5 = 0.15 * (x1**2 - x2**2)
    
    y = term1 + term2 + term3 + term4 + term5
    return X, y

def prepare_dataset():
    X, y = generate_data(3000, seed=42)
    X_train, y_train = X[:2000], y[:2000]
    X_val, y_val = X[2000:2500], y[2000:2500]
    X_test, y_test = X[2500:], y[2500:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def train_model(hidden_sizes, train_data, val_data, epochs=300, lr=0.05, batch_size=64):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task='regression', seed=42)
    
    train_losses = []
    val_losses = []
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            grads = model.backward(X_batch, y_batch)
            model.update(grads, lr)
            
        train_pred = model.forward(X_train)
        val_pred = model.forward(X_val)
        
        train_loss = compute_mse(train_pred, y_train)
        val_loss = compute_mse(val_pred, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    return model, train_losses, val_losses

def plot_results(model_2l, model_3l, history_2l, history_3l, test_data):
    os.makedirs('../figures', exist_ok=True)
    
    # 1. 繪製 Loss 曲線圖 (loss_curves.png)
    plt.figure(figsize=(10, 5))
    plt.plot(history_2l[0], label='2-Layer Train Loss', color='blue', linestyle='--')
    plt.plot(history_2l[1], label='2-Layer Val Loss', color='blue')
    plt.plot(history_3l[0], label='3-Layer Train Loss', color='orange', linestyle='--')
    plt.plot(history_3l[1], label='3-Layer Val Loss', color='orange')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Log scale)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('../figures/loss_curves.png', dpi=300)
    plt.close()
    print("✅ Loss 曲線圖已儲存至 hw01/figures/loss_curves.png")

    # 2. 繪製曲面比較圖 (regression_comparison.png)
    X_test, y_test = test_data
    y_pred_2l = model_2l.forward(X_test)
    y_pred_3l = model_3l.forward(X_test)

    fig = plt.figure(figsize=(16, 4))
    
    # 網格化以繪製 3D 或散布對比
    # 這裡以測試集散布圖或切片展示，為了符合直覺我們展示預測值與真實值對比
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(X_test[:, 0], X_test[:, 1], y_test.flatten(), c=y_test.flatten(), cmap='viridis', s=10)
    ax1.set_title('True Surface')
    
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.scatter(X_test[:, 0], X_test[:, 1], y_pred_2l.flatten(), c=y_pred_2l.flatten(), cmap='viridis', s=10)
    ax2.set_title('2-Layer Prediction')
    
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3.scatter(X_test[:, 0], X_test[:, 1], y_pred_3l.flatten(), c=y_pred_3l.flatten(), cmap='viridis', s=10)
    ax3.set_title('3-Layer Prediction')
    
    ax4 = fig.add_subplot(1, 4, 4)
    errors_2l = np.abs(y_pred_2l - y_test)
    errors_3l = np.abs(y_pred_3l - y_test)
    ax4.hist(errors_2l, bins=30, alpha=0.5, label='2-Layer Error', color='blue')
    ax4.hist(errors_3l, bins=30, alpha=0.5, label='3-Layer Error', color='orange')
    ax4.set_title('Absolute Error Distribution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/regression_comparison.png', dpi=300)
    plt.close()
    print("✅ 曲面與誤差比較圖已儲存至 hw01/figures/regression_comparison.png")

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_dataset()
    
    print("正在訓練 2 層 MLP ([32])...")
    model_2l, train_loss_2, val_loss_2 = train_model([32], train_data, val_data, epochs=300, lr=0.05)
    
    print("正在訓練 3 層 MLP ([32, 32])...")
    model_3l, train_loss_3, val_loss_3 = train_model([32, 32], train_data, val_data, epochs=300, lr=0.05)
    
    # 測試集評估
    test_mse_2l = compute_mse(model_2l.forward(test_data[0]), test_data[1])
    test_mse_3l = compute_mse(model_3l.forward(test_data[0]), test_data[1])
    
    print(f"\n[測試結果]")
    print(f"2-Layer MLP 測試集 MSE: {test_mse_2l:.5f}")
    print(f"3-Layer MLP 測試集 MSE: {test_mse_3l:.5f}")
    
    # 產出圖表
    plot_results(model_2l, model_3l, (train_loss_2, val_loss_2), (train_loss_3, val_loss_3), test_data)