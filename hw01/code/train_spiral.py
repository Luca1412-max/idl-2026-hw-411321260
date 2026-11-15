# hw01/code/train_spiral.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP
from generate_data import generate_spiral_data

def prepare_spiral_datasets():
    # 嚴格依老師回饋：noise=0.035，資料種子 42，驗證 100，測試 200
    X_train, y_train = generate_spiral_data(num_samples_per_class=1000, noise=0.035, seed=42)
    X_val, y_val = generate_spiral_data(num_samples_per_class=250, noise=0.035, seed=100)
    X_test, y_test = generate_spiral_data(num_samples_per_class=250, noise=0.035, seed=200)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_bce_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_classification_model(hidden_sizes, train_data, val_data, epochs=400, lr=0.1, batch_size=64):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    model = MLP(input_dim=2, hidden_sizes=hidden_sizes, output_dim=1, task='classification', seed=42)
    train_losses, val_losses = [], []
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
            
        train_pred = model.forward(X_train)
        val_pred = model.forward(X_val)
        train_losses.append(compute_bce_loss(train_pred, y_train))
        val_losses.append(compute_bce_loss(val_pred, y_val))
            
    return model, train_losses, val_losses

def plot_spiral_boundaries(model_2l, model_3l, test_data):
    os.makedirs('../figures', exist_ok=True)
    X_test, y_test = test_data
    
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z_2l = model_2l.forward(grid_points).reshape(xx.shape)
    Z_3l = model_3l.forward(grid_points).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, Z, title in zip(axes, [Z_2l, Z_3l], ['2-Layer MLP ([32])', '3-Layer MLP ([32, 32])']):
        ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
        ax.scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1], c='blue', label='Class 0', edgecolor='k', s=20)
        ax.scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1], c='red', label='Class 1', edgecolor='k', s=20)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        ax.set_title(f'{title} Decision Boundary (noise=0.035)')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('../figures/spiral_boundaries.png', dpi=300)
    plt.close()
    print("✅ 雙螺旋決策邊界圖已更新至 hw01/figures/spiral_boundaries.png")

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_spiral_datasets()
    
    print("正在訓練 2 層分類模型...")
    model_2l, train_l_2, val_l_2 = train_classification_model([32], train_data, val_data, epochs=400, lr=0.1)
    
    print("正在訓練 3 層分類模型...")
    model_3l, train_l_3, val_l_3 = train_classification_model([32, 32], train_data, val_data, epochs=400, lr=0.1)
    
    # 測試集實際評估與輸出數字
    test_pred_2l = model_2l.forward(test_data[0])
    test_pred_3l = model_3l.forward(test_data[0])
    
    acc_2l = np.mean((test_pred_2l >= 0.5) == test_data[1])
    acc_3l = np.mean((test_pred_3l >= 0.5) == test_data[1])
    
    y_true = test_data[1].astype(int)
    tp_2 = np.sum(((test_pred_2l >= 0.5) == 1) & (y_true == 1))
    tn_2 = np.sum(((test_pred_2l >= 0.5) == 0) & (y_true == 0))
    fp_2 = np.sum(((test_pred_2l >= 0.5) == 1) & (y_true == 0))
    fn_2 = np.sum(((test_pred_2l >= 0.5) == 0) & (y_true == 1))
    
    tp_3 = np.sum(((test_pred_3l >= 0.5) == 1) & (y_true == 1))
    tn_3 = np.sum(((test_pred_3l >= 0.5) == 0) & (y_true == 0))
    fp_3 = np.sum(((test_pred_3l >= 0.5) == 1) & (y_true == 0))
    fn_3 = np.sum(((test_pred_3l >= 0.5) == 0) & (y_true == 1))
    
    print(f"\n[實際輸出結果 - 雙螺旋分類]")
    print(f"2-Layer MLP Accuracy: {acc_2l*100:.2f}% | CM: TP={tp_2}, TN={tn_2}, FP={fp_2}, FN={fn_2}")
    print(f"3-Layer MLP Accuracy: {acc_3l*100:.2f}% | CM: TP={tp_3}, TN={tn_3}, FP={fp_3}, FN={fn_3}")
    
    plot_spiral_boundaries(model_2l, model_3l, test_data)