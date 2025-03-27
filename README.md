# Steg Solver

The problem is as follows. We want to generate a weight vector w such that sgn(w^Tx_i) = y_i where x_i is the image data (flattened out) and y_i is \pm 1, where 1 represents the presense of steg, and -1 represents an unedited photo.

We consider the loss function sum_i exp(-y_iw^Tx_i).

We have the data in `data/X.txt` and `data/y.txt`. There are 1000 cover images and 1000 stego images, each with 136 features. The data has been standardized (eg the average of all the iamges has been removed and the results scaled) and there is now one constant feature so that no matrix is all zeros.

## How to Run

You should have `numpy`, `tdqm`, `matplotlib` installed, simply because I wanted to actually see what the training times looked like quickly. Then run `main.py` any way you like. If you want to test it, there is a test flag inside of `main`, which runs each one for only a few iterations.

Note that each one should run for the entirety of what we ask it for, as I wanted to make them run for longer instead of having an early stopping criterion. I don't believe this data has a perfect discriminator, so I wanted to see how good of a solution I could get.

## Report

So I just used python and numpy because it was the fastest one for me to program rather than the fastest to actually solve it. I think I'd want to try implementing this in Rust with ndarray, but that's for another day.

The problem asks just for Newton's method and for Gradient Descent. Those are relatively easy to implement, as we have precomputed the Hessian and gradient of the loss function.

However, there are variations of each method:

- Newton/Quasi-Newton Methods
  - Newton's Method
  - BFGS
  - L-BFGS
- Gradient Descent Methods
  - Gradient Descent
  - Conjugate Gradient Descent
  - Stochastic Gradient Descent
  - Minibatch Gradient Descent

Note that in the code that these aren't ordered in this way. I ordered them by the order of implementation.

So I implemented these as well. The rest are the results from these methods. I wanted a way to compute the similarities, and as w is just a weight vector, I used the cosine similarity.

## Results

### Performance Comparison

#### Convergence Speed

- Newton's method achieved 90% accuracy in just 1 iteration (107.00ms)
- Other methods' convergence to 90% accuracy:
  - BFGS: 74 iterations (76.87ms)
  - L-BFGS: 202 iterations (139.10ms)
  - Conjugate Gradient: 209 iterations (153.89ms)
  - Gradient Descent: 581 iterations (592.28ms)
  - Minibatch GD: 615 iterations (674.66ms)
  - SGD: Did not reach 90% accuracy
- Final accuracies:
  - Conjugate Gradient: 95.00% (best)
  - Newton: 94.80%
  - BFGS: 94.65%
  - L-BFGS: 94.65%
  - Gradient Descent: 94.45%
  - Minibatch GD: 94.25%
  - SGD: 85.60%

#### Computational Efficiency

Average time per iteration:

- SGD: 0.21ms (fastest)
- L-BFGS: 0.74ms
- Gradient Descent: 0.98ms
- BFGS: 1.11ms
- Conjugate Gradient: 1.11ms
- Minibatch GD: 1.17ms
- Newton: 51.64ms (slowest)

#### Loss Values

Final loss values:

- Newton: 461.24
- BFGS: 461.55
- L-BFGS: 461.70
- Conjugate Gradient: 479.84
- Gradient Descent: 552.02
- Minibatch GD: 581.73
- SGD: 1654.82 (highest)

### Visualization Analysis

![view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/training_progress.png)

The training progress plots ([view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/training_progress.png)) show:
- Newton's method achieves rapid convergence in both loss and accuracy
- Other methods show slower but steady convergence patterns
- SGD exhibits more volatile behavior and slower convergence
- All methods except SGD eventually achieve similar final accuracies

![view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/timing_comparison.png)

The timing comparison ([view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/timing_comparison.png)) reveals:
- Newton's method has significantly higher per-iteration cost
- Other methods have comparable per-iteration costs
- Time to 90% accuracy varies significantly between methods
- BFGS shows the best balance of convergence speed and per-iteration cost

![view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/similarity_matrix.png)

The similarity matrix ([view figure](https://github.com/agniv-the-marker/a-linear-classifier-with-exponential-loss/blob/main/figures/similarity_matrix.png)) shows:
- Newton's method solution is notably different from other methods
- Other methods show varying degrees of similarity to each other
- Quasi-Newton methods (BFGS, L-BFGS) show high similarity to each other

### Key Observations

1. Newton's method achieves rapid convergence but at a higher computational cost per iteration
2. Stochastic methods (SGD) show more volatile behavior and slower convergence
3. Quasi-Newton methods (BFGS, L-BFGS) achieve good balance of speed and accuracy
4. Conjugate Gradient achieves the highest final accuracy
5. All methods except SGD eventually converge to similar high accuracies (>94%)
6. The solution space appears to have multiple local minima, as evidenced by the different final solutions

## Unexpectedness

So, one thing that really surprised me is the orthogonality of the solution from Newton's method and every single other one. I'm not too surprised with SGD being slower to converge, as it is a tradeoff we have to make, but I believe it should eventually get to the solution that the other ones attained.

I think this surprises me because Newton's method almost instantly gets above 90%, which to me implies that the other methods should have headed in a similar direction. So, in theory, tracking similarity over time makes sense to see how quickly it diverges?
