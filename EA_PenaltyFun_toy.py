import numpy as np
import matplotlib.pyplot as plt

# Define the objective function to be minimized
def f(x):
    return (x - 1) ** 2

# Define the constraint function (inequality constraint g(x) ≤ 0)
def g(x):
    return x - 1.5

# Define the penalty function method for constrained optimization
def pfm(x, k, c):
    """
    Penalty function method for handling constraints in optimization problems.
    
    Parameters:
    x : float or ndarray
        Input variable(s) to evaluate
    k : float
        Penalty coefficient (controls the strength of penalty)
    c : float
        Threshold value for the inequality constraint (typically 0)
        
    Returns:
    float or ndarray
        The penalized objective function value
    """
    # Calculate constraint violation (how much the constraint is violated)
    h = c - g(x)
    # Return original function plus penalty term (only applied when h < 0, meaning constraint is violated)，
    # ‘max(h,0)’ can also be used here instead of np.maximum(h, 0).
    return f(x) + k * np.maximum(h, 0)

# Generate X values for plotting (from -5 to 5 with step size 0.1)
X = np.arange(-5, 5.1, 0.1)

# Calculate the penalty function values with:
# k = 10 (penalty coefficient)
# c = 0 (constraint threshold, meaning we enforce g(x) ≤ 0)
Y = pfm(X, k=10, c=0)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X, Y, label="Penalty Function Method (k=10, c=0)", color="blue")
plt.xlabel("X")
plt.ylabel("pfm(X, k, c)")
plt.title("Penalty Function Method")
plt.grid(True)
plt.legend()
plt.show()