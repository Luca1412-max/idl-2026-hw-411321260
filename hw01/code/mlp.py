# hw01/code/mlp.py
import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_sizes, output_dim, task='regression', seed=42):
        np.random.seed(seed)
        self.task = task
        self.layers_dims = [input_dim] + hidden_sizes + [output_dim]
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layers_dims) - 1):
            fan_in = self.layers_dims[i]
            # 老師建議：tanh 隱藏層改用 Xavier 初始化 (np.sqrt(1.0 / fan_in))
            limit = np.sqrt(1.0 / fan_in)
            w = np.random.uniform(-limit, limit, (self.layers_dims[i], self.layers_dims[i+1]))
            b = np.zeros((1, self.layers_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current_input = np.tanh(z)
            self.activations.append(current_input)
            
        # 輸出層
        z_out = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        
        if self.task == 'regression':
            out = z_out # 線性輸出
        else:
            # 分類使用 Sigmoid，加上數值穩定限制
            out = 1.0 / (1.0 + np.exp(-np.clip(z_out, -500, 500)))
        self.activations.append(out)
        return out

    def predict(self, X):
        """老師要求的 predict() 方法"""
        out = self.forward(X)
        if self.task == 'classification':
            return (out >= 0.5).astype(int)
        return out

    def backward(self, X, y):
        m = X.shape[0]
        out = self.forward(X)
        
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        
        # 輸出層 delta (移除不必要的 clip)
        if self.task == 'regression':
            delta = (out - y) / m
        else:
            delta = (out - y) / m
            
        grads_w[-1] = np.dot(self.activations[-2].T, delta)
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)
        
        # 隱藏層 delta
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * (1.0 - self.activations[i+1] ** 2)
            grads_w[i] = np.dot(self.activations[i].T, delta)
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            
        return {'weights': grads_w, 'biases': grads_b}

    def update(self, grads, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads['weights'][i]
            self.biases[i] -= lr * grads['biases'][i]