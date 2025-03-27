import numpy as np
from tqdm import tqdm
import time
from data_loader import loss_function, gradient, calculate_accuracy

def hessian(w, X, y):
    """Compute the Hessian matrix"""
    predictions = X @ w
    exp_terms = np.exp(-y * predictions)
    H = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(y)):
        x_i = X[i:i+1].T
        H += exp_terms[i] * x_i @ x_i.T
    return H

def newton_solver(X, y, max_iter=1000, tol=1e-8, verbose=False):
    """Solve for w using Newton's method"""
    w = np.zeros(X.shape[1])
    losses = []
    accuracies = []
    times = []
    weights_history = [w.copy()]
    
    iterator = tqdm(range(max_iter), desc="Newton's Method")
    
    for i in iterator:
        start_time = time.time()
        
        current_loss = loss_function(w, X, y)
        losses.append(current_loss)
        _, _, current_acc = calculate_accuracy(w, X, y)
        accuracies.append(current_acc)
        
        iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
        
        grad = gradient(w, X, y)
        H = hessian(w, X, y)
        
        delta = np.linalg.solve(H, grad)
        w_new = w - delta
        
        weights_history.append(w_new.copy())
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if np.linalg.norm(w_new - w) < tol:
            if verbose:
                print(f"\nNewton converged after {i} iterations")
            break
            
        w = w_new
    
    return w, losses, accuracies, times, weights_history

def bfgs_solver(X, y, max_iter=1000, tol=1e-8, initial_lr=0.001, verbose=False):
    """Solve for w using BFGS method"""
    n = X.shape[1]
    w = np.zeros(n)
    H = np.eye(n)
    grad = gradient(w, X, y)
    losses = []
    accuracies = []
    times = []
    weights_history = [w.copy()]
    
    iterator = tqdm(range(max_iter), desc="BFGS")
    
    for i in iterator:
        try:
            start_time = time.time()
            current_loss = loss_function(w, X, y)
            losses.append(current_loss)
            _, _, current_acc = calculate_accuracy(w, X, y)
            accuracies.append(current_acc)
            
            iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
            
            d = -H @ grad
            d_norm = np.linalg.norm(d)
            if d_norm > 1e3:
                d = d / d_norm
            
            lr = initial_lr
            max_attempts = 20
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    w_new = w + lr * d
                    new_loss = loss_function(w_new, X, y)
                    
                    if new_loss < current_loss:
                        break
                    lr *= 0.5
                    attempts += 1
                except (RuntimeWarning, OverflowError):
                    lr *= 0.1
                    attempts += 1
                    continue
            
            if attempts == max_attempts:
                if verbose:
                    print(f"\nBFGS stopped due to line search failure after {i} iterations")
                return w, losses, accuracies, times, weights_history
            
            grad_new = gradient(w_new, X, y)
            s = w_new - w
            y_diff = grad_new - grad
            
            try:
                rho = 1.0 / np.dot(s, y_diff)
                I = np.eye(n)
                H = ((I - rho * np.outer(s, y_diff)) @ H @ 
                     (I - rho * np.outer(y_diff, s)) + 
                     rho * np.outer(s, s))
                
            except (np.linalg.LinAlgError, RuntimeWarning):
                if verbose:
                    print(f"\nBFGS skipping update due to numerical error at iteration {i}")
                continue
            
            if np.linalg.norm(w_new - w) < tol:
                if verbose:
                    print(f"\nBFGS converged after {i} iterations")
                break
            
            w = w_new
            grad = grad_new
            weights_history.append(w.copy())
            end_time = time.time()
            times.append(end_time - start_time)
            
        except (RuntimeWarning, OverflowError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"\nBFGS stopped due to numerical error after {i} iterations: {str(e)}")
            return w, losses, accuracies, times, weights_history
    
    return w, losses, accuracies, times, weights_history

def lbfgs_solver(X, y, max_iter=1000, tol=1e-8, initial_lr=0.001, m=10, verbose=False):
    """Solve for w using L-BFGS method"""
    n_features = X.shape[1]
    w = np.zeros(n_features)
    grad = gradient(w, X, y)
    
    s_list = []  # Storage for position differences
    diff_list = []  # Storage for gradient differences
    rho_list = []  # Storage for rho values
    
    losses = []
    accuracies = []
    times = []
    weights_history = [w.copy()]
    
    iterator = tqdm(range(max_iter), desc="L-BFGS")
    
    for i in iterator:
        try:
            start_time = time.time()
            current_loss = loss_function(w, X, y)
            losses.append(current_loss)
            _, _, current_acc = calculate_accuracy(w, X, y)
            accuracies.append(current_acc)
            
            iterator.set_postfix_str(f'loss: {current_loss:.4f}, acc: {current_acc:.1f}%')
            
            # Two-loop recursion
            q = grad.copy()
            alphas = []
            
            for s, diff, rho in zip(reversed(s_list), reversed(diff_list), reversed(rho_list)):
                alpha = rho * np.dot(s, q)
                alphas.append(alpha)
                q = q - alpha * diff
            
            # Initial Hessian approximation
            if len(s_list) > 0:
                gamma = np.dot(s_list[-1], diff_list[-1]) / np.dot(diff_list[-1], diff_list[-1])
                gamma = np.clip(gamma, 1e-3, 1e3)
            else:
                gamma = 1.0
            
            r = gamma * q
            
            for s, diff, rho, alpha in zip(s_list, diff_list, rho_list, reversed(alphas)):
                beta = rho * np.dot(diff, r)
                r = r + (alpha - beta) * s
            
            d = -r
            
            if np.linalg.norm(d) < 1e-8:
                if verbose:
                    print(f"\nL-BFGS stopping due to small search direction at iteration {i}")
                break
            
            # Line search
            lr = initial_lr
            found_better = False
            max_attempts = 20
            attempts = 0
            
            while attempts < max_attempts:
                try:
                    w_new = w + lr * d
                    new_loss = loss_function(w_new, X, y)
                    
                    if new_loss < current_loss:
                        found_better = True
                        break
                    lr *= 0.5
                    attempts += 1
                except (RuntimeWarning, OverflowError):
                    lr *= 0.1
                    attempts += 1
                    continue
            
            if not found_better:
                if verbose:
                    print(f"\nL-BFGS stopped due to line search failure at iteration {i}")
                break
            
            grad_new = gradient(w_new, X, y)
            s = w_new - w
            grad_diff = grad_new - grad
            
            sy = np.dot(s, grad_diff)
            if sy > 1e-8:
                rho = 1.0 / sy
                s_list.append(s)
                diff_list.append(grad_diff)
                rho_list.append(rho)
                
                if len(s_list) > m:
                    s_list.pop(0)
                    diff_list.pop(0)
                    rho_list.pop(0)
            
            if np.linalg.norm(w_new - w) < tol:
                if verbose:
                    print(f"\nL-BFGS converged after {i} iterations")
                break
            
            w = w_new
            grad = grad_new
            weights_history.append(w.copy())
            end_time = time.time()
            times.append(end_time - start_time)
            
        except (RuntimeWarning, OverflowError) as e:
            if verbose:
                print(f"\nL-BFGS warning at iteration {i}: {str(e)}")
            continue
    
    return w, losses, accuracies, times, weights_history 