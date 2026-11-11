# hw01/code/train_spiral.py
import numpy as np

def generate_spiral_data(num_samples_per_class=1000, noise=0.2, seed=42):
    """
    生成雙螺旋二元分類資料集
    每個類別各 num_samples_per_class 筆
    """
    np.random.seed(seed)
    N = num_samples_per_class
    
    # 類別 0 (k=0)
    t0 = np.random.uniform(0, 1, N)
    r0 = 0.12 + 0.82 * t0
    theta0 = 3.0 * np.pi * t0 + 0.0
    x1_0 = r0 * np.cos(theta0) + np.random.normal(0, noise, N)
    x2_0 = r0 * np.sin(theta0) + np.random.normal(0, noise, N)
    class0 = np.stack([x1_0, x2_0, np.zeros(N)], axis=1)
    
    # 類別 1 (k=1)
    t1 = np.random.uniform(0, 1, N)
    r1 = 0.12 + 0.82 * t1
    theta1 = 3.0 * np.pi * t1 + np.pi
    x1_1 = r1 * np.cos(theta1) + np.random.normal(0, noise, N)
    x2_1 = r1 * np.sin(theta1) + np.random.normal(0, noise, N)
    class1 = np.stack([x1_1, x2_1, np.ones(N)], axis=1)
    
    # 合併並洗牌
    data = np.vstack([class0, class1])
    np.random.shuffle(data)
    
    X = data[:, :2]
    y = data[:, 2:3]
    return X, y

def prepare_spiral_datasets():
    # 總共需要：訓練 2000 筆 (每類 1000)、驗證 500 筆 (每類 250)、測試 500 筆 (每類 250)
    X_train, y_train = generate_spiral_data(num_samples_per_class=1000, seed=42)
    X_val, y_val = generate_spiral_data(num_samples_per_class=250, seed=100)
    X_test, y_test = generate_spiral_data(num_samples_per_class=250, seed=200)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_spiral_datasets()
    print(f"訓練集大小: {train_data[0].shape}, 類別分佈: {np.bincount(train_data[1].flatten().astype(int))}")
    print(f"驗證集大小: {val_data[0].shape}, 類別分佈: {np.bincount(val_data[1].flatten().astype(int))}")
    print(f"測試集大小: {test_data[0].shape}, 類別分佈: {np.bincount(test_data[1].flatten().astype(int))}")