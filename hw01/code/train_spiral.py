# hw01/code/train_spiral.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP

def generate_spiral_data(num_samples_per_class=1000, noise=0.2, seed=42):
    np.random.seed(seed)
    N = num_samples_per_class
    
    t0 = np.random.uniform(0, 1, N)
    r0 = 0.12 + 0.82 * t0
    theta0 = 3.0 * np.pi * t0 + 0.0
    x1_0 = r0 * np.cos(theta0) + np.random.normal(0, noise, N)
    x2_0 = r0 * np.sin(theta0) + np.random.normal(0, noise, N)
    class0 = np.stack([x1_0, x2_0, np.zeros(N)], axis=1)
    
    t1 = np.random.uniform(0, 1, N)
    r1 = 0.12 + 0.82 * t1
    theta1 = 3.0 * np.pi * t1 + np.pi
    x1_1 = r1 * np.cos(theta1) + np.random.normal(0, noise, N)
    x2_1 = r1 * np.sin(theta1) + np.random.normal(0, noise, N)
    class1 = np.stack([x1_1, x2_1, np.ones(N)], axis=1)
    
    data = np.vstack([class0, class1])
    np.random.shuffle(data)
    
    X = data[:, :2]
    y = data[:, 2:3]
    return X, y

def prepare_spiral_datasets():
    X_train, y_train = generate_spiral_data(num_samples_per_class=1000, seed=42)
    X_val, y_val = generate_spiral_data(num_samples_per_class=250, seed=100)
    X_test, y_test = generate_spiral_data(num_samples_per_class=250, seed=200)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def compute_bce_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_accuracy(y_pred, y_true):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    return np.mean(y_pred_labels == y_true)

def compute_confusion_matrix(y_pred, y_true):
    y_pred_labels = (y_pred >= 0.5).astype(int)
    y_true = y_true.astype(int)
    tp = np.sum((y_pred_labels == 1) & (y_true == 1))
    tn = np.sum((y_pred_labels == 0) & (y_true == 0))
    fp = np.sum((y_pred_labels == 1) & (y_true == 0))
    fn = np.sum((y_pred_labels == 0) & (y_true == 1))
    return tp, tn, fp, fn

def train_classification_model(hidden_sizes, train_data, val_data, epochs=400, lr=0.1, batch_size=64):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task='classification', seed=42)
    
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
        
        train_losses.append(compute_bce_loss(train_pred, y_train))
        val_losses.append(compute_bce_loss(val_pred, y_val))
            
    return model, train_losses, val_losses

def plot_decision_boundaries(model_2l, model_3l, test_data):
    os.makedirs('../figures', exist_ok=True)
    X_test, y_test = test_data
    
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z_2l = model_2l.forward(grid_points).reshape(xx.shape)
    Z_3l = model_3l.forward(grid_points).reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2-Layer Model
    ax = axes[0]
    ax.contourf(xx, yy, Z_2l, levels=50, cmap='RdBu', alpha=0.6)
    ax.scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1], c='blue', label='Class 0', edgecolor='k', s=20)
    ax.scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1], c='red', label='Class 1', edgecolor='k', s=20)
    ax.contour(xx, yy, Z_2l, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax.set_title('2-Layer MLP ([32]) Decision Boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    
    # 3-Layer Model
    ax = axes[1]
    ax.contourf(xx, yy, Z_3l, levels=50, cmap='RdBu', alpha=0.6)
    ax.scatter(X_test[y_test.flatten() == 0, 0], X_test[y_test.flatten() == 0, 1], c='blue', label='Class 0', edgecolor='k', s=20)
    ax.scatter(X_test[y_test.flatten() == 1, 0], X_test[y_test.flatten() == 1, 1], c='red', label='Class 1', edgecolor='k', s=20)
    ax.contour(xx, yy, Z_3l, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    ax.set_title('3-Layer MLP ([32, 32]) Decision Boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/spiral_boundaries.png', dpi=300)
    plt.close()
    print("✅ 雙螺旋決策邊界圖已成功儲存至 hw01/figures/spiral_boundaries.png")

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_spiral_datasets()
    
    print("正在訓練 2 層 MLP 分類模型 ([32])...")
    model_2l, train_l_2, val_l_2 = train_classification_model([32], train_data, val_data, epochs=400, lr=0.1)
    
    print("正在訓練 3 層 MLP 分類模型 ([32, 32])...")
    model_3l, train_l_3, val_l_3 = train_classification_model([32, 32], train_data, val_data, epochs=400, lr=0.1)
    
    # 測試集評估
    test_pred_2l = model_2l.forward(test_data[0])
    test_pred_3l = model_3l.forward(test_data[0])
    
    acc_2l = compute_accuracy(test_pred_2l, test_data[1])
    acc_3l = compute_accuracy(test_pred_3l, test_data[1])
    tp_2, tn_2, fp_2, fn_2 = compute_confusion_matrix(test_pred_2l, test_data[1])
    tp_3, tn_3, fp_3, fn_3 = compute_confusion_matrix(test_pred_3l, test_data[1])
    
    print(f"\n[測試結果]")
    print(f"2-Layer MLP 測試集 Accuracy: {acc_2l*100:.2f}% | Confusion Matrix: TP={tp_2}, TN={tn_2}, FP={fp_2}, FN={fn_2}")
    print(f"3-Layer MLP 測試集 Accuracy: {acc_3l*100:.2f}% | Confusion Matrix: TP={tp_3}, TN={tn_3}, FP={fp_3}, FN={fn_3}")
    
    # 繪製決策邊界
    plot_decision_boundaries(model_2l, model_3l, test_data)