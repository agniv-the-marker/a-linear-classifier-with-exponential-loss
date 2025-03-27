import numpy as np
import matplotlib.pyplot as plt

def find_accuracy_threshold(accuracies, threshold=90):
    """Find the first iteration where accuracy exceeds the threshold"""
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i
    return None

def plot_training_progress(losses_dict, accuracies_dict, threshold_points=None):
    """Plot training progress for all methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.set_title('Loss vs Iterations')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    for method, losses in losses_dict.items():
        ax1.plot(losses, label=method)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies with threshold points
    ax2.set_title('Accuracy vs Iterations')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy (%)')
    for method, accs in accuracies_dict.items():
        line, = ax2.plot(accs, label=method)
        if threshold_points and method in threshold_points and threshold_points[method] is not None:
            idx = threshold_points[method]
            ax2.plot(idx, accs[idx], 'o', color=line.get_color(), 
                    markersize=10, label=f'{method} 90%')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_timing_comparison(timing_dict):
    """Plot average time per step for each method"""
    plt.figure(figsize=(10, 5))
    methods = list(timing_dict.keys())
    times = [timing_dict[method] * 1000 for method in methods]  # Convert to milliseconds
    
    plt.bar(methods, times)
    plt.title('Average Time per Step')
    plt.ylabel('Time (milliseconds)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def compute_similarity(w1, w2):
    """Compute cosine similarity between two weight vectors"""
    cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    return cos_sim

def compute_all_pairwise_similarities(weights_dict, verbose=False):
    """Compute similarities between all pairs of methods"""
    methods = list(weights_dict.keys())
    n = len(methods)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            sim = compute_similarity(weights_dict[methods[i]], weights_dict[methods[j]])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            if verbose and i != j:
                print(f"{methods[i]} vs {methods[j]}: {sim:.4f}")
    
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, methods):
    """Plot similarity matrix between all methods"""
    n = len(methods)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    plt.xticks(range(n), methods, rotation=45, ha='right')
    plt.yticks(range(n), methods)
    
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                    ha='center', va='center',
                    color='black' if abs(similarity_matrix[i, j]) < 0.5 else 'white')
    
    plt.title('Solution Similarity Matrix')
    plt.tight_layout()
    plt.show() 