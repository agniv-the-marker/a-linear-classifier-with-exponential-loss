import numpy as np

def load_data(X_path, y_path):
    """Load data from X.txt and y.txt files"""
    X = np.loadtxt(X_path)
    y = np.loadtxt(y_path)
    
    print(f"Loaded data shapes - X: {X.shape}, y: {y.shape}")
    
    if X.shape[0] != len(y):
        raise ValueError(f"Dimension mismatch: X has {X.shape[0]} samples but y has {len(y)} samples")
    
    return X, y

def loss_function(w, X, y):
    """Compute the loss function l(w) = sum(exp(-y_i * w^T * x_i))"""
    assert X.shape[0] == len(y), "Dimension mismatch: X has {} samples but y has {} samples".format(X.shape[0], len(y))
    predictions = X @ w
    return np.sum(np.exp(-y * predictions))

def gradient(w, X, y):
    """Compute the gradient of the loss function"""
    predictions = X @ w
    exp_terms = np.exp(-y * predictions)
    return -X.T @ (y * exp_terms)

def calculate_accuracy(w, X, y):
    """Calculate the accuracy of the model"""
    predictions = np.sign(X @ w)
    correct = np.sum(predictions == y)
    total = len(y)
    return correct, total, (correct/total) * 100 