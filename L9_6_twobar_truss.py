import numpy as np
import matplotlib.pyplot as plt

def MOEAD(max_gen, n_subproblems, n_neighbors, mutation_rate):
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D).    
    Args:
        max_gen (int): Maximum number of generations.
        n_subproblems (int): Number of subproblems (population size).
        n_neighbors (int): Size of the neighborhood for each subproblem.
        mutation_rate (float): Probability of mutation for each variable.   
    Returns:
        population (ndarray): Final population of solutions.
        fitness (ndarray): Fitness values of the final population.
    """
    # Initialization
    dim = 3  # Number of decision variables
    x_min = np.array([1e-5, 1e-5, 1])  # Lower bounds for the variables
    x_max = np.array([100, 100, 3])    # Upper bounds for the variables

    # Initialize weight vectors (lambdas) for subproblems
    lambdas = np.zeros((n_subproblems, 2))
    lambdas[:, 0] = np.linspace(0, 1, n_subproblems)  # Weight for first objective
    lambdas[:, 1] = np.linspace(1, 0, n_subproblems)  # Weight for second objective

    # Initialize the neighborhood of the subproblems based on weight vectors
    neighbourhood = np.zeros((n_subproblems, n_neighbors), dtype=int)
    for i in range(n_subproblems):
        dist = np.sum((lambdas[i, :] - lambdas) ** 2, axis=1)  # Calculate Euclidean distances
        index = np.argsort(dist)  # Sort to find nearest neighbors
        neighbourhood[i, :] = index[:n_neighbors]  # Store indices of nearest neighbors

    # Initialize a random population within the bounds
    population = x_min + np.random.rand(n_subproblems, dim) * (x_max - x_min)

    # Evaluate the fitness of the initial population
    fitness, objs, cv1, cv2, cv3 = two_bar_truss_obj(population)

    # Initialize the reference point (ideal point)
    z = np.min(fitness, axis=0)  # Best observed values for each objective

    # Evolution loop over generations
    for gen in range(max_gen):
        offsprings = population.copy()  # Create a copy of the current population
        for i in range(n_subproblems):
            # Randomly select two parent indices from neighborhood
            j = neighbourhood[i, np.random.randint(n_neighbors)]
            k = neighbourhood[i, np.random.randint(n_neighbors)]

            # Perform crossover: combine genes from two parents
            idx = np.random.rand(dim) > 0.5  # Random crossover points
            offsprings[i, :] = population[j, :]
            offsprings[i, idx] = population[k, idx]

            # Perform mutation: introduce random change to offspring
            idx = np.random.rand(dim) < mutation_rate  # Apply mutation based on rate
            genes = x_min + np.random.rand(dim) * (x_max - x_min)  # Random genes
            offsprings[i, idx] = genes[idx]  # Mutate selected genes

        # Evaluate the fitness of the offspring
        fitness_new, objs, cv1, cv2, cv3 = two_bar_truss_obj(offsprings)

        # Update the reference point z
        z = np.minimum(z, np.min(fitness_new, axis=0))

        # Update the neighboring solutions with new offspring
        for i in range(n_subproblems):
            lambda_ = lambdas[i, :]  # Weight vector for current subproblem
            g = Tchebycheff(fitness[i, :], lambda_, z)  # Current Tchebycheff value
            for j in range(n_neighbors):
                neighbour = neighbourhood[i, j]
                h = Tchebycheff(fitness_new[neighbour, :], lambda_, z)  # New Tchebycheff value
                if h < g:  # If offspring is better, replace the current solution
                    g = h
                    population[i, :] = offsprings[neighbour, :]
                    fitness[i, :] = fitness_new[neighbour, :]

        # Print progress
        f_min = np.min(fitness, axis=0)  # Best fitness values in current generation
        print(f'gen: {gen + 1}, {f_min[0]:.2f}, {f_min[1]:.2f}')

    return population, fitness

def two_bar_truss_obj(x):
    """
    Evaluate both objective functions and constraints for the two-bar truss problem.
    
    Args:
        x (ndarray): Population of solutions (each row is a solution).
    
    Returns:
        F (ndarray): Penalized and scaled objective values.
        obj (ndarray): Raw objective values.
        g1 (ndarray): Constraint violation for volume.
        g2 (ndarray): Constraint violation for stress on AC.
        g3 (ndarray): Constraint violation for stress on BC.
    """
    # Evaluate both objective functions and constraints for the population
    f1 = obj1(x)  # Objective 1: structural volume
    f2 = obj2(x)  # Objective 2: displacement of the joint

    g1 = np.maximum(const1(x), 0)  # Constraint 1: total volume <= 0.1
    g2 = np.maximum(const2(x), 0)  # Constraint 2: stress to AC <= 10^5
    g3 = np.maximum(const3(x), 0)  # Constraint 3: stress to BC <= 10^5

    # Penalty coefficients for combining objectives with constraints
    k11 = 250      # Penalty coefficient for g1 in f1
    k21 = 0.125    # Penalty coefficient for g2 in f1
    k31 = 2.5e-2   # Penalty coefficient for g3 in f1
    k12 = 2.5e6    # Penalty coefficient for g1 in f2
    k22 = 1250     # Penalty coefficient for g2 in f2
    k32 = 250      # Penalty coefficient for g3 in f2

    obj = np.column_stack((f1, f2))  # Stack the raw objectives
    # Add penalties to objectives
    f1 += k11 * g1 + k21 * g2 + k31 * g3
    f2 += k12 * g1 + k22 * g2 + k32 * g3
    
    # Scale objectives for normalization
    f1 *= 100      # Scale first objective for better visualization
    f2 /= 1000     # Scale second objective for better visualization
    F = np.column_stack((f1, f2))

    return F, obj, g1, g2, g3

def obj1(x):
    """
    Compute structural volume (Objective 1).   
    Args:
        x (ndarray): Population of solutions.    
    Returns:
        ndarray: Structural volume for each solution.
    """
    x1 = x[:, 0]  # Cross-sectional area of bar AC
    x2 = x[:, 1]  # Cross-sectional area of bar BC
    x3 = x[:, 2]  # Length parameter
    return x1 * np.sqrt(16 + x3**2) + x2 * np.sqrt(1 + x3**2)

def obj2(x):
    """
    Compute displacement of the joint (Objective 2).
    Args:
        x (ndarray): Population of solutions.
    Returns:
        ndarray: Displacement for each solution.
    """
    x1 = x[:, 0]  # Cross-sectional area of bar AC
    x3 = x[:, 2]  # Length parameter
    return 20 * np.sqrt(16 + x3**2) / (x1 * x3)

def const1(x):
    """
    Constraint 1: Total volume must not exceed 0.1.   
    Args:
        x (ndarray): Population of solutions.    
    Returns:
        ndarray: Constraint violation values.
    """
    return obj1(x) - 0.1  # Positive if constraint is violated

def const2(x):
    """
    Constraint 2: Stress on AC must not exceed 10^5.   
    Args:
        x (ndarray): Population of solutions.    
    Returns:
        ndarray: Constraint violation values.
    """
    return obj2(x) - 1e5  # Positive if constraint is violated

def const3(x):
    """
    Constraint 3: Stress on BC must not exceed 10^5.    
    Args:
        x (ndarray): Population of solutions.   
    Returns:
        ndarray: Constraint violation values.
    """
    x2 = x[:, 1]  # Cross-sectional area of bar BC
    x3 = x[:, 2]  # Length parameter
    return 80 * np.sqrt(1 + x3**2) / (x2 * x3) - 1e5  # Positive if constraint is violated

def Tchebycheff(fitness, lambda_, ideal):
    """
    Compute the Tchebycheff aggregation value for a given solution.
    
    Args:
        fitness (ndarray): Fitness values of the solution.
        lambda_ (ndarray): Weight vector for the subproblem.
        ideal (ndarray): Reference point (ideal values).
    
    Returns:
        float: Tchebycheff aggregation value.
    """
    return np.max(np.abs(fitness - ideal) * lambda_)  # Weighted Chebyshev distance

max_gen = 300  # Maximum number of generations
num_subproblems = 400  # Total number of subproblems (population size)
neighbourhood_size = 20  # Size of the neighborhood for each subproblem
mutation_rate = 0.2  # Mutation rate for genetic algorithm

# Run MOEA/D to solve the two-bar truss problem
population, fitness = MOEAD(max_gen, num_subproblems, neighbourhood_size, mutation_rate)

# Display the resulting Pareto front
plt.figure()
plt.scatter(fitness[:, 0], fitness[:, 1], marker='o')
plt.title('Best solutions in the objective space (the approximated Pareto front)')
plt.xlabel('f1 * 100')  # First objective scaled by 100
plt.ylabel('f2 / 1000')  # Second objective divided by 1000
plt.show()