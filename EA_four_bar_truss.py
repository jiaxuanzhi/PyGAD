import numpy as np
import matplotlib.pyplot as plt

def four_bar_truss_obj(x):
    """Objective function for the four-bar truss problem.
    Returns two objectives: total volume (f1) and displacement (f2)."""
    L = 200  # Length parameter
    F = 10   # Force parameter
    E = 2 * 1e5  # Elastic modulus
    sqrt2 = np.sqrt(2)  # Square root of 2
    sqrt22 = 2 * sqrt2  # 2 times square root of 2
    # First objective: total volume of the truss (to be minimized)
    f1 = L * (2 * x[:, 0] + sqrt2 * x[:, 1] + np.sqrt(x[:, 2]) + x[:, 3])
    # Second objective: displacement (to be minimized)
    f2 = F * L / E * (2 / x[:, 0] + sqrt22 / x[:, 1] - sqrt22 / x[:, 2] + 2 / x[:, 3])
    # Scale f1 for better numerical behavior
    f1 = f1 / 1e4
    # Return both objectives as a 2-column array
    return np.column_stack((f1, f2))

def Tchebycheff(fitness, lambda_, ideal):
    """Tchebycheff decomposition approach for scalarizing multi-objective problems."""
    return np.max(np.abs(fitness - ideal) * lambda_)

def MOEAD(max_gen, n_subproblems, n_neighbors, mutation_rate):
    """MOEA/D algorithm for multi-objective optimization."""
    # Initialization
    dim = 4  # dimension of the design variable
    F = 10  # Problem parameter
    sigma = 10  # Problem parameter
    factor = F / sigma  # Scaling factor
    sqrt2 = np.sqrt(2)  # Square root of 2
    
    # Define bounds for each design variable
    x_min = np.array([factor, sqrt2 * factor, sqrt2 * factor, factor])  # lower bounds
    x_max = np.array([3 * factor, 3 * factor, 3 * factor, 3 * factor])  # upper bounds

    # Initialize weight vectors for the subproblems (evenly distributed)
    lambdas = np.zeros((n_subproblems, 2))
    lambdas[:, 0] = np.linspace(0, 1, n_subproblems)  # weights for first objective
    lambdas[:, 1] = np.linspace(1, 0, n_subproblems)  # weights for second objective

    # Initialize the neighbourhood of each subproblem (closest weight vectors)
    neighbourhood = np.zeros((n_subproblems, n_neighbors), dtype=int)
    for i in range(n_subproblems):
        # Calculate Euclidean distance between weight vectors
        dist = np.sum((lambdas[i, :] - lambdas) ** 2, axis=1)
        # Get indices of closest neighbors
        neighbourhood[i, :] = np.argsort(dist)[:n_neighbors]

    # Initialize population randomly within bounds
    population = x_min + np.random.rand(n_subproblems, dim) * (x_max - x_min)

    # Evaluate initial fitness
    fitness = four_bar_truss_obj(population)

    # Initialize the reference point (ideal point) with best found values
    z = np.min(fitness, axis=0)

    # Main optimization loop
    for gen in range(max_gen):
        # Create offspring population
        offsprings = population.copy()
        
        # Generate new solutions for each subproblem
        for i in range(n_subproblems):
            # Select two random neighbors
            j = neighbourhood[i, np.random.randint(n_neighbors)]
            k = neighbourhood[i, np.random.randint(n_neighbors)]
            
            # Crossover: mix genes from two parents
            idx = np.random.rand(dim) > 0.5  # random mask for crossover
            offsprings[i, idx] = population[k, idx]  # take some genes from parent k
            
            # Mutation: randomly change some genes
            idx = np.random.rand(dim) < mutation_rate  # mutation mask
            genes = x_min + np.random.rand(dim) * (x_max - x_min)  # random values
            offsprings[i, idx] = genes[idx]  # apply mutation

        # Evaluate new solutions
        fitness_new = four_bar_truss_obj(offsprings)

        # Update reference point (ideal point)
        z = np.minimum(z, np.min(fitness_new, axis=0))

        # Update neighboring solutions if improvement found
        for i in range(n_subproblems):
            lambda_ = lambdas[i, :]  # weight vector for this subproblem
            g = Tchebycheff(fitness[i, :], lambda_, z)  # current scalarized fitness
            
            # Check all neighbors for better solutions
            for j in range(n_neighbors):
                neighbour = neighbourhood[i, j]
                h = Tchebycheff(fitness_new[neighbour, :], lambda_, z)  # new scalarized fitness
                if h < g:  # if improvement found
                    g = h
                    population[i, :] = offsprings[neighbour, :]  # update solution
                    fitness[i, :] = fitness_new[neighbour, :]  # update fitness

        # Print progress
        f_min = np.min(fitness, axis=0)  # current best objectives
        print(f'gen: {gen}, {f_min[0]:.2f}, {f_min[1]:.2f}')

    return population, fitness

# Algorithm parameters
max_gen = 250  # maximum number of generations
num_subproblems = 100  # number of subproblems/weight vectors
neighbourhood_size = 20  # size of neighborhood for each subproblem
mutation_rate = 0.1  # probability of mutation for each gene

# Run MOEA/D optimization
population, fitness = MOEAD(max_gen, num_subproblems, neighbourhood_size, mutation_rate)

# Plot the results (approximated Pareto front)
plt.scatter(fitness[:, 0], fitness[:, 1], marker='o')
plt.title('Best solutions in the objective space (the approximated Pareto front)')
plt.xlabel('f1 (scaled volume)')
plt.ylabel('f2 (displacement)')
plt.show()