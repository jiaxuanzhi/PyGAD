import numpy as np
import matplotlib.pyplot as plt

def MOEAD(x_min, x_max, dim, max_gen, n_subproblems, n_neighbors, mutation_rate):
    """
    Implementation of the MOEA/D algorithm for multi-objective optimization (ZDT1 problem).

    Parameters:
    - x_min: Minimum value for each dimension of the solution
    - x_max: Maximum value for each dimension of the solution
    - dim: Number of dimensions of the problem
    - max_gen: Maximum number of generations (iterations)
    - n_subproblems: Number of subproblems (population size)
    - n_neighbors: Size of the neighborhood for each subproblem
    - mutation_rate: Probability of mutation for each gene

    Returns:
    - population: Final population of solutions
    - fitness: Objective values of the final population
    """
    # Initialize preference vectors (weights) for each subproblem
    lambdas = np.zeros((n_subproblems, 2))  # Each row is a weight vector for a subproblem
    lambdas[:, 0] = np.linspace(0, 1, n_subproblems)  # Weight for the first objective
    lambdas[:, 1] = np.linspace(1, 0, n_subproblems)  # Weight for the second objective

    # Initialize the neighborhood for each subproblem
    neighbourhood = np.zeros((n_subproblems, n_neighbors), dtype=int)
    for i in range(n_subproblems):
        # Calculate the Euclidean distance between subproblem i and all others
        dist = np.sum((lambdas[i, :] - lambdas) ** 2, axis=1)
        # Sort subproblems by distance and select the closest n_neighbors
        index = np.argsort(dist) # Indices of subproblems sorted by distance
        neighbourhood[i, :] = index[:n_neighbors] # Nearest subproblems

    # Initialize a random population within the bounds [x_min, x_max]
    population = x_min + np.random.rand(n_subproblems, dim) * (x_max - x_min)
    # Evaluate the initial population using the ZDT1 function
    fitness = ZDT1(population)

    # Initialize the reference point z as the best-known minimum for each objective
    z = np.min(fitness, axis=0)
    # Main optimization loop
    for gen in range(max_gen):
        offsprings = population.copy()  # Create a copy of the population for offspring
        for i in range(n_subproblems):
            # Randomly select two parents from the neighborhood of subproblem i
            parents = np.random.choice(neighbourhood[i, :], 2, replace=False)
            j, k = parents[0], parents[1] # Indices of the selected parents
            # Crossover: Create an offspring by combining genes from parents j and k
            idx = np.random.rand(dim) > 0.5  # Randomly select genes 
            offsprings[i, :] = population[j, :] # Initialize the offspring with genes from parent j
            offsprings[i, idx] = population[k, idx] # Replace some genes with genes from parent k
            # Mutation: Randomly mutate some genes in the offspring
            idx = np.random.rand(dim) < mutation_rate  # Randomly select genes to mutate
            # Mutate the selected genes to a random value within the bounds
            offsprings[i, idx] = x_min + np.random.rand(np.sum(idx)) * (x_max - x_min)  

        # Evaluate the offspring population using the ZDT1 function
        fitness_new = ZDT1(offsprings)
        # Update the reference point z with the new minimum values
        z = np.minimum(z, np.min(fitness_new, axis=0))

        # Update the population by replacing solutions in the neighborhood
        for i in range(n_subproblems):
            lambda_ = lambdas[i, :]  # Weight vector for subproblem i
            g = Tchebycheff(fitness[i, :], lambda_, z)  # Decomposition value of the current solution
            for j in range(n_neighbors):
                neighbour = neighbourhood[i, j]  # Neighboring subproblem j
                h = Tchebycheff(fitness_new[neighbour, :], lambda_, z)  # Decomposition value of the neighbor
                if h < g:  # If the neighbor is better, replace the current solution
                    g = h # Update the decomposition value
                    population[i, :] = offsprings[neighbour, :] # Replace the solution
                    fitness[i, :] = fitness_new[neighbour, :] # Replace the fitness value

        # Print the progress of the optimization
        print(f"Generation {gen + 1}, Min Fitness: {np.min(fitness, axis=0)}")

    return population, fitness

def Tchebycheff(fitness, lambda_, ideal):
    """
    Calculate the Tchebycheff decomposition value for a solution.
    Parameters:
    - fitness: Objective values of the solution
    - lambda_: Weight vector for the subproblem
    - ideal: Reference point (best-known minimum for each objective)
    Returns:
    - Decomposition value
    """
    return np.max(np.abs(fitness - ideal) * lambda_)

def ZDT1(x):
    """
    Calculate the objective values for the ZDT1 problem.
    Parameters:
    - x: Population of solutions (each row is a solution)
    Returns:
    - Objective values (each row contains the objective values for a solution)
    """
    output = np.zeros((x.shape[0], 2))  # Initialize the output matrix
    output[:, 0] = x[:, 0]  # First objective value
    g = 1 + 9 / (x.shape[1] - 1) * np.sum(x[:, 1:], axis=1)  # Calculate g(x)
    output[:, 1] = g * (1 - np.sqrt(x[:, 0] / g))  # Second objective value
    return output

# Parameters for the MOEA/D algorithm
x_min = 0  # Minimum value for each dimension
x_max = 1  # Maximum value for each dimension
dim = 30  # Number of dimensions
max_gen = 1000  # Maximum number of generations
n_subproblems = 100  # Number of subproblems (population size)
n_neighbors = 20  # Neighborhood size
mutation_rate = 0.03  # Mutation rate

# Run the MOEA/D algorithm to optimize the ZDT1 problem
population, fitness = MOEAD(x_min, x_max, dim, max_gen, n_subproblems, n_neighbors, mutation_rate)

# Plot the approximate Pareto front
plt.scatter(fitness[:, 0], fitness[:, 1], marker='o')  # Scatter plot of the solutions
plt.xlim([0, 1])  # Set the x-axis range
plt.ylim([0, 1])  # Set the y-axis range
plt.xlabel('Objective 1')  # Label for the x-axis
plt.ylabel('Objective 2')  # Label for the y-axis
plt.title('Approximate Pareto Front')  # Title of the plot
plt.show()  # Display the plot