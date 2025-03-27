import numpy as np
from tqdm import tqdm
import time
from data_loader import loss_function, gradient, calculate_accuracy

def gradient_descent_solver(X, y, initial_lr=0.001, max_iter=5000, tol=1e-8, verbose=False):
    """Solve for w using gradient descent with line search"""
    w = np.zeros(X.shape[1])
    losses = []
    accuracies = []
    times = []
    
    iterator = tqdm(range(max_iter), desc="Gradient Descent")
    
    for i in iterator:
        start_time = time.time()
        current_loss = loss_function(w, X, y)
        losses.append(current_loss)
        _, _, current_acc = calculate_accuracy(w, X, y)
        accuracies.append(current_acc)
        
        iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
        
        grad = gradient(w, X, y)
        
        lr = initial_lr
        while True:
            w_new = w - lr * grad
            new_loss = loss_function(w_new, X, y)
            
            if new_loss < current_loss:
                break
            lr *= 0.5
            if lr < 1e-10:
                if verbose:
                    print(f"\nGD stopped due to tiny learning rate after {i} iterations")
                return w, losses, accuracies, np.mean(times)
        
        if np.linalg.norm(w_new - w) < tol:
            if verbose:
                print(f"\nGD converged after {i} iterations")
            break
            
        w = w_new
        end_time = time.time()
        times.append(end_time - start_time)
    
    return w, losses, accuracies, np.mean(times)

def conjugate_gradient_solver(X, y, max_iter=5000, tol=1e-14, verbose=False, initial_lr=0.001):
    """Solve for w using conjugate gradient descent with backtracking line search"""
    w = np.zeros(X.shape[1])
    grad = gradient(w, X, y)
    d = -grad  # Initial direction is negative gradient
    losses = []
    accuracies = []
    times = []
    
    iterator = tqdm(range(max_iter), desc="Conjugate Gradient")
    
    for i in iterator:
        start_time = time.time()
        current_loss = loss_function(w, X, y)
        losses.append(current_loss)
        _, _, current_acc = calculate_accuracy(w, X, y)
        accuracies.append(current_acc)
        
        iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
        
        d_norm = np.linalg.norm(d)
        if d_norm > 1e3:
            d = d / d_norm
        
        lr = initial_lr
        min_lr = 1e-14
        found_better = False
        
        while lr > min_lr:
            w_new = w + lr * d
            new_loss = loss_function(w_new, X, y)
            
            if new_loss < current_loss:
                found_better = True
                break
            lr *= 0.5
        
        if not found_better:
            lr = min_lr  # Use minimum learning rate if no better point found
            w_new = w + lr * d
            if verbose:
                print(f"\nCGD using minimum learning rate at iteration {i}")
        
        grad_new = gradient(w_new, X, y)
        
        grad_norm = np.dot(grad, grad)
        beta = max(0, np.dot(grad_new, grad_new - grad) / grad_norm)
        beta = min(beta, 1.0)
        
        d = -grad_new + beta * d
        
        if np.linalg.norm(w_new - w) < tol:
            if verbose:
                print(f"\nCGD converged after {i} iterations")
            break
        
        w = w_new
        grad = grad_new
        end_time = time.time()
        times.append(end_time - start_time)
    
    return w, losses, accuracies, np.mean(times)

def stochastic_gradient_descent_solver(X, y, batch_size=32, initial_lr=0.001, max_iter=10000, tol=1e-8, verbose=False):
    """Solve for w using stochastic gradient descent"""
    n_samples = X.shape[0]
    w = np.zeros(X.shape[1])
    losses = []
    accuracies = []
    times = []
    
    initial_lr = initial_lr / batch_size
    
    iterator = tqdm(range(max_iter), desc="SGD")
    
    for i in iterator:
        start_time = time.time()
        
        try:
            current_loss = loss_function(w, X, y)
            losses.append(current_loss)
            _, _, current_acc = calculate_accuracy(w, X, y)
            accuracies.append(current_acc)
            
            iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
            
            indices = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            predictions = X_batch @ w
            predictions = np.clip(predictions, -100, 100)
            
            exp_terms = np.exp(-y_batch * predictions)
            grad = -(X_batch.T @ (y_batch * exp_terms))
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e3:
                grad = grad * (1e3 / grad_norm)
            
            lr = initial_lr / (1 + 0.001 * i)
            
            w_new = w - lr * grad
            
            if np.linalg.norm(w_new - w) < tol:
                if verbose:
                    print(f"\nSGD converged after {i} iterations")
                break
            
            w = w_new
            
        except (RuntimeWarning, OverflowError) as e:
            if verbose:
                print(f"\nWarning at iteration {i}: {str(e)}")
            continue
            
        end_time = time.time()
        times.append(end_time - start_time)
    
    return w, losses, accuracies, np.mean(times)

def minibatch_gradient_descent_solver(X, y, batch_size=100, initial_lr=0.001, max_iter=10000, tol=1e-8, verbose=False):
    """Solve for w using minibatch gradient descent"""
    n_samples = X.shape[0]
    w = np.zeros(X.shape[1])
    losses = []
    accuracies = []
    times = []
    
    n_batches = n_samples // batch_size
    
    iterator = tqdm(range(max_iter), desc="Minibatch GD")
    
    for i in iterator:
        start_time = time.time()
        
        try:
            current_loss = loss_function(w, X, y)
            losses.append(current_loss)
            _, _, current_acc = calculate_accuracy(w, X, y)
            accuracies.append(current_acc)
            
            iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
            
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            grad_sum = np.zeros_like(w)
            for j in range(n_batches):
                start_idx = j * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                predictions = X_batch @ w
                predictions = np.clip(predictions, -100, 100)
                
                exp_terms = np.exp(-y_batch * predictions)
                grad_batch = -(X_batch.T @ (y_batch * exp_terms))
                
                grad_sum += grad_batch
            
            grad = grad_sum / n_batches
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e3:
                grad = grad * (1e3 / grad_norm)
            
            lr = initial_lr
            found_better = False
            
            while lr > 1e-10:
                w_new = w - lr * grad
                new_loss = loss_function(w_new, X, y)
                
                if new_loss < current_loss:
                    found_better = True
                    break
                lr *= 0.5
            
            if not found_better:
                if verbose:
                    print(f"\nMinibatch GD stopped due to line search failure at iteration {i}")
                break
            
            if np.linalg.norm(w_new - w) < tol:
                if verbose:
                    print(f"\nMinibatch GD converged after {i} iterations")
                break
            
            w = w_new
            
        except (RuntimeWarning, OverflowError) as e:
            if verbose:
                print(f"\nWarning at iteration {i}: {str(e)}")
            continue
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return w, losses, accuracies, np.mean(times) 