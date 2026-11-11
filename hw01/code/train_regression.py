# hw01/code/train_regression.py
import numpy as np
import os
import matplotlib.pyplot as plt
from mlp import MLP

def generate_data(num_samples=3000, seed=42):
    """
    生成雙變數非線性函數資料
    x1, x2 in [-1, 1]
    """
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(num_samples, 2))
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    
    # 題目指定的複雜非線性函數
    term1 = 0.45 * np.sin(2.5 * np.pi * x1) * np.cos(1.5 * np.pi * x2)
    term2 = 0.25 * np.sin(3.0 * np.pi * x1 * x2)
    term3 = 0.20 * np.exp(-8.0 * ((x1 - 0.35)**2 + (x2 + 0.25)**2))
    term4 = -0.20 * np.exp(-10.0 * ((x1 + 0.40)**2 + (x2 - 0.35)**2))
    term5 = 0.15 * (x1**2 - x2**2)
    
    y = term1 + term2 + term3 + term4 + term5
    return X, y

def prepare_dataset():
    # 總共 3000 筆：訓練集 2000，驗證集 500，測試集 500
    X, y = generate_data(3000, seed=42)
    
    X_train, y_train = X[:2000], y[:2000]
    X_val, y_val = X[2000:2500], y[2000:2500]
    X_test, y_test = X[2500:], y[2500:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_dataset()
    print(f"訓練集大小: {train_data[0].shape}, {train_data[1].shape}")
    print(f"驗證集大小: {val_data[0].shape}, {val_data[1].shape}")
    print(f"測試集大小: {test_data[0].shape}, {test_data[1].shape}")