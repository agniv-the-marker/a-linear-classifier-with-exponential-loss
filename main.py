from data_loader import load_data
from newton_methods import newton_solver, bfgs_solver, lbfgs_solver
from gradient_methods import (
    gradient_descent_solver, 
    conjugate_gradient_solver,
    stochastic_gradient_descent_solver, 
    minibatch_gradient_descent_solver
)
from analysis_utils import (
    find_accuracy_threshold,
    plot_training_progress,
    plot_timing_comparison,
    compute_all_pairwise_similarities,
    plot_similarity_matrix
)

def main():
    verbose_training = False
    verbose_similarity = False
    testing = False
    
    # Load data
    X, y = load_data('./data/X.txt', './data/y.txt')
    print(f"Data dimensions: X: {X.shape}, y: {y.shape}")
    
    # Set iterations based on testing flag
    newton_iters = 1000 if not testing else 20
    other_iters = 10000 if not testing else 20
    sgd_iters = 100000 if not testing else 20
    
    # Run all solvers
    w_newton, losses_newton, accs_newton, time_newton = newton_solver(
        X, y, max_iter=newton_iters, tol=1e-12, verbose=verbose_training)
    
    w_gd, losses_gd, accs_gd, time_gd = gradient_descent_solver(
        X, y, max_iter=other_iters, tol=1e-8, verbose=verbose_training)
    
    w_cgd, losses_cgd, accs_cgd, time_cgd = conjugate_gradient_solver(
        X, y, max_iter=other_iters, tol=1e-14, verbose=verbose_training, initial_lr=0.001)
    
    w_bfgs, losses_bfgs, accs_bfgs, time_bfgs = bfgs_solver(
        X, y, max_iter=other_iters, tol=1e-8, initial_lr=0.001, verbose=verbose_training)
    
    w_sgd, losses_sgd, accs_sgd, time_sgd = stochastic_gradient_descent_solver(
        X, y, batch_size=20, max_iter=sgd_iters, tol=1e-8, verbose=verbose_training)
    
    w_mbgd, losses_mbgd, accs_mbgd, time_mbgd = minibatch_gradient_descent_solver(
        X, y, batch_size=100, max_iter=other_iters, tol=1e-8, verbose=verbose_training)
    
    w_lbfgs, losses_lbfgs, accs_lbfgs, time_lbfgs = lbfgs_solver(
        X, y, max_iter=other_iters, tol=1e-8, initial_lr=0.001, m=10, verbose=verbose_training)
    
    # Find points where accuracy exceeds 90%
    threshold_points = {
        "Newton": find_accuracy_threshold(accs_newton),
        "Gradient Descent": find_accuracy_threshold(accs_gd),
        "Conjugate Gradient": find_accuracy_threshold(accs_cgd),
        "BFGS": find_accuracy_threshold(accs_bfgs),
        "SGD": find_accuracy_threshold(accs_sgd),
        "Minibatch GD": find_accuracy_threshold(accs_mbgd),
        "L-BFGS": find_accuracy_threshold(accs_lbfgs)
    }
    
    # Create dictionaries for plotting
    losses_dict = {
        "Newton": losses_newton,
        "Gradient Descent": losses_gd,
        "Conjugate Gradient": losses_cgd,
        "BFGS": losses_bfgs,
        "SGD": losses_sgd,
        "Minibatch GD": losses_mbgd,
        "L-BFGS": losses_lbfgs
    }
    
    accuracies_dict = {
        "Newton": accs_newton,
        "Gradient Descent": accs_gd,
        "Conjugate Gradient": accs_cgd,
        "BFGS": accs_bfgs,
        "SGD": accs_sgd,
        "Minibatch GD": accs_mbgd,
        "L-BFGS": accs_lbfgs
    }
    
    timing_dict = {
        "Newton": time_newton,
        "Gradient Descent": time_gd,
        "Conjugate Gradient": time_cgd,
        "BFGS": time_bfgs,
        "SGD": time_sgd,
        "Minibatch GD": time_mbgd,
        "L-BFGS": time_lbfgs
    }
    
    weights_dict = {
        "Newton": w_newton,
        "Gradient Descent": w_gd,
        "Conjugate Gradient": w_cgd,
        "BFGS": w_bfgs,
        "SGD": w_sgd,
        "Minibatch GD": w_mbgd,
        "L-BFGS": w_lbfgs
    }
    
    # Plot results
    plot_training_progress(losses_dict, accuracies_dict, threshold_points)
    plot_timing_comparison(timing_dict)
    
    # Compute and plot similarities
    similarity_matrix = compute_all_pairwise_similarities(weights_dict, verbose=verbose_similarity)
    plot_similarity_matrix(similarity_matrix, list(weights_dict.keys()))
    
    # Print summary statistics
    print("\nIterations to reach 90% accuracy:")
    for method, point in threshold_points.items():
        if point is not None:
            print(f"{method}: {point} iterations")
        else:
            print(f"{method}: Did not reach 90% accuracy")
    
    print("\nFinal loss values:")
    for method, losses in losses_dict.items():
        print(f"{method}: {losses[-1]:.4f}")
    
    print("\nFinal accuracies:")
    for method, accs in accuracies_dict.items():
        print(f"{method}: {accs[-1]:.2f}%")

if __name__ == "__main__":
    main()
