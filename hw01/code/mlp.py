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
            # He initialization: std = sqrt(2 / fan_in)
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def _tanh(self, x):
        return np.tanh(x)

    def _sigmoid(self, x):
        # 數值穩定版 sigmoid
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向傳播
        X 形状: (m, input_dim)
        """
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        for i in range(self.num_layers):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # 隱藏層使用 tanh，輸出層根據任務決定
            if i == self.num_layers - 1:
                if self.task == 'regression':
                    a = z  # 線性輸出
                elif self.task == 'classification':
                    a = self._sigmoid(z)  # 二元分類用 sigmoid
                else:
                    raise ValueError(f"Unknown task: {self.task}")
            else:
                a = self._tanh(z)
                
            self.activations.append(a)
            current_input = a
            
        return self.activations[-1]