import numpy as np
import matplotlib.pyplot as plt
import os

def find_accuracy_threshold(accuracies, times, threshold=90):
    """Find the first iteration where accuracy exceeds the threshold and its corresponding time"""
    cumulative_time = np.cumsum(times)
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            return i, cumulative_time[i]
    return None, None

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
        if threshold_points and method in threshold_points:
            point = threshold_points[method]
            if point is not None:
                idx, _ = point
                if idx is not None:  # Check that idx is not None
                    ax2.plot(idx, accs[idx], 'o', color=line.get_color(), 
                            markersize=10, label=f'{method} 90%')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join('figures', 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_timing_comparison(timing_dict, threshold_points):
    """Plot average time per step and time to 90% accuracy for each method"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot average time per step
    methods = list(timing_dict.keys())
    times = [np.mean(timing_dict[method]) * 1000 for method in methods]  # Convert to milliseconds
    
    ax1.bar(methods, times)
    ax1.set_title('Average Time per Step')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y')
    
    # Plot time to 90% accuracy
    times_to_threshold = []
    for method in methods:
        if threshold_points[method] is not None:
            _, time = threshold_points[method]
            if time is not None:
                times_to_threshold.append(time * 1000)  # Convert to milliseconds
            else:
                times_to_threshold.append(np.nan)
        else:
            times_to_threshold.append(np.nan)
    
    ax2.bar(methods, times_to_threshold)
    ax2.set_title('Time to 90% Accuracy')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join('figures', 'timing_comparison.png'), dpi=300, bbox_inches='tight')
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
    # Save the figure
    plt.savefig(os.path.join('figures', 'similarity_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_newton_similarities(weights_history_dict):
    """
    Plot similarities between Newton's method and other methods over iterations.
    
    Args:
        weights_history_dict: Dictionary containing weight history for each method
    """
    methods = list(weights_history_dict.keys())
    newton_weights = weights_history_dict["Newton"]
    n_iterations = len(newton_weights)
    
    # Calculate similarities over time
    similarities = {}
    for method in methods:
        if method != "Newton":
            method_weights = weights_history_dict[method]
            # Take the minimum length to handle different iteration counts
            min_iterations = min(len(method_weights), n_iterations)
            similarities[method] = [
                compute_similarity(newton_weights[i], method_weights[i])
                for i in range(min_iterations)
            ]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    for method, sims in similarities.items():
        plt.plot(sims, label=method)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cosine Similarity with Newton\'s Method')
    plt.title('Solution Similarity to Newton\'s Method Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join('figures', 'newton_similarities.png'), dpi=300, bbox_inches='tight')
    plt.show()