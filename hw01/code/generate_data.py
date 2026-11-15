# hw01/code/generate_data.py
import numpy as np

def generate_regression_data(num_samples=3000, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(num_samples, 2))
    x1, x2 = X[:, 0:1], X[:, 1:2]
    
    term1 = 0.45 * np.sin(2.5 * np.pi * x1) * np.cos(1.5 * np.pi * x2)
    term2 = 0.25 * np.sin(3.0 * np.pi * x1 * x2)
    term3 = 0.20 * np.exp(-8.0 * ((x1 - 0.35)**2 + (x2 + 0.25)**2))
    term4 = -0.20 * np.exp(-10.0 * ((x1 + 0.40)**2 + (x2 - 0.35)**2))
    term5 = 0.15 * (x1**2 - x2**2)
    y = term1 + term2 + term3 + term4 + term5
    return X, y

def generate_spiral_data(num_samples_per_class=1000, noise=0.035, seed=42):
    """依照老師回饋將 noise 改為 0.035"""
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
    x2_1 = r1 * np.sin(noise if False else 0, noise, N) if False else r1 * np.sin(theta1) + np.random.normal(0, noise, N)
    class1 = np.stack([x1_1, x2_1, np.ones(N)], axis=1)
    
    data = np.vstack([class0, class1])
    np.random.shuffle(data)
    return data[:, :2], data[:, 2:3]