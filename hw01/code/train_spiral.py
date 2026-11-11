# hw01/code/train_spiral.py
import numpy as np
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

def train_classification_model(hidden_sizes, train_data, val_data, epochs=400, lr=0.1, batch_size=64):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, task='classification', seed=42)
    
    train_losses = []
    val_losses = []
    val_accs = []
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
        
        train_loss = compute_bce_loss(train_pred, y_train)
        val_loss = compute_bce_loss(val_pred, y_val)
        val_acc = compute_accuracy(val_pred, y_val)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
    return model, train_losses, val_losses, val_accs

if __name__ == "__main__":
    train_data, val_data, test_data = prepare_spiral_datasets()
    
    print("\n--- 訓練兩層 MLP 分類模型 ([32]) ---")
    model_2l, train_l_2, val_l_2, val_acc_2 = train_classification_model([32], train_data, val_data, epochs=300, lr=0.1)
    
    print("\n--- 訓練三層 MLP 分類模型 ([32, 32]) ---")
    model_3l, train_l_3, val_l_3, val_acc_3 = train_classification_model([32, 32], train_data, val_data, epochs=300, lr=0.1)