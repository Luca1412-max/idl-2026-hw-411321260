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

def train_model(hidden_sizes, train_data, val_data, epochs=300, lr=0.01, batch_size=64):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task='regression', seed=42)
    
    train_losses = []
    val_losses = []
    
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Mini-batch 訓練洗牌
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward & Backward
            grads = model.backward(X_batch, y_batch)
            model.update(grads, lr)
            
        # 計算每個 epoch 結束後的訓練與驗證 MSE
        train_pred = model.forward(X_train)
        val_pred = model.forward(X_val)
        
        train_loss = compute_mse(train_pred, y_train)
        val_loss = compute_mse(val_pred, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")
            
    return model, train_losses, val_losses

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_dataset()
    
    print("\n--- 訓練兩層 MLP ([32]) ---")
    model_2layer, train_loss_2, val_loss_2 = train_model([32], train_data, val_data, epochs=200, lr=0.05)
    
    print("\n--- 訓練三層 MLP ([32, 32]) ---")
    model_3layer, train_loss_3, val_loss_3 = train_model([32, 32], train_data, val_data, epochs=200, lr=0.05)