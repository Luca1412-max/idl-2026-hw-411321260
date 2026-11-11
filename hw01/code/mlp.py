# hw01/code/mlp.py
import numpy as np

class MLP:
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int, task: str = 'classification', seed: int = 42):
        np.random.seed(seed)
        self.task = task
        self.layers_dim = [input_dim] + hidden_sizes + [output_dim]
        self.num_layers = len(self.layers_dim) - 1
        
        # 初始化權重與偏置 (使用 He Initialization 適合 tanh)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            fan_in = self.layers_dim[i]
            fan_out = self.layers_dim[i+1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, a):
        # 導數可以用 activation 運算：1 - a^2
        return 1.0 - a ** 2

    def _sigmoid(self, x):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向傳播
        X 形狀: (m, input_dim)
        """
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        for i in range(self.num_layers):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == self.num_layers - 1:
                if self.task == 'regression':
                    a = z  # 迴歸線性輸出
                elif self.task == 'classification':
                    a = self._sigmoid(z)  # 分類用 sigmoid
                else:
                    raise ValueError(f"Unknown task: {self.task}")
            else:
                a = self._tanh(z)
                
            self.activations.append(a)
            current_input = a
            
        return self.activations[-1]

    def backward(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        反向傳播：計算權重與偏置的梯度
        X: (m, input_dim)
        y: (m, output_dim)
        """
        m = X.shape[0]
        # 先執行一次 forward 確保 activations 與 z_values 正確
        self.forward(X)
        
        grads = {'weights': [], 'biases': []}
        delta = None
        
        # 輸出層梯度計算
        y_pred = self.activations[-1]
        if self.task == 'regression':
            # MSE 損失對 z 的導數 (假設輸出為線性): (y_pred - y) / m
            delta = (y_pred - y) / m
        elif self.task == 'classification':
            # BCE 損失搭配 Sigmoid 的組合導數簡化為: (y_pred - y) / m
            # 為了數值穩定加入極小值避免除以零
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            delta = (y_pred_clipped - y) / m
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        # 從後往前反向傳播
        layer_grads_w = [None] * self.num_layers
        layer_grads_b = [None] * self.num_layers
        
        for i in range(self.num_layers - 1, -1, -1):
            a_prev = self.activations[i]
            layer_grads_w[i] = np.dot(a_prev.T, delta)
            layer_grads_b[i] = np.sum(delta, axis=0, keepdims=True)
            
            if i > 0:
                # 傳遞 delta 到前一層
                activation_derivative = self._tanh_derivative(self.activations[i])
                delta = np.dot(delta, self.weights[i].T) * activation_derivative
                
        grads['weights'] = layer_grads_w
        grads['biases'] = layer_grads_b
        return grads

    def update(self, grads: dict, lr: float):
        """
        使用梯度下降更新權重與偏置
        """
        for i in range(self.num_layers):
            self.weights[i] -= lr * grads['weights'][i]
            self.biases[i] -= lr * grads['biases'][i]