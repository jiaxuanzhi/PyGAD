import numpy as np
import matplotlib.pyplot as plt

def MOEAD(max_gen, n_subproblems, n_neighbors, mutation_rate):
    """MOEA/D algorithm for multi-objective optimization."""
    dim = 3  # dimension of the design variable
    
    # Define bounds for each design variable
    x_min = np.array([0.01, 0.01, 1.0])  # lower bounds
    x_max = np.array([100.0, 100.0, 3.0])  # upper bounds

    # Initialize weight vectors for the subproblems (evenly distributed)
    lambdas = np.zeros((n_subproblems, 2))
    lambdas[:, 0] = np.linspace(0, 1, n_subproblems)  # weights for first objective
    lambdas[:, 1] = np.linspace(1, 0, n_subproblems)  # weights for second objective

    # Initialize the neighbourhood of each subproblem (closest weight vectors)
    neighbourhood = np.zeros((n_subproblems, n_neighbors), dtype=int)
    for i in range(n_subproblems):
        dist = np.sum((lambdas[i, :] - lambdas) ** 2, axis=1)
        neighbourhood[i, :] = np.argsort(dist)[:n_neighbors]

    # Initialize population randomly within bounds
    population = x_min + np.random.rand(n_subproblems, dim) * (x_max - x_min)

    # Evaluate initial fitness
    fitness = truss_obj_with_penalty(population)

    # Initialize the reference point (ideal point) with best found values
    z = np.min(fitness, axis=0)

    for gen in range(max_gen):
        offsprings = population.copy()
        
        for i in range(n_subproblems):
            j = neighbourhood[i, np.random.randint(n_neighbors)]
            k = neighbourhood[i, np.random.randint(n_neighbors)]
            
            # Crossover: mix genes from two parents
            idx = np.random.rand(dim) > 0.5
            offsprings[i, idx] = population[k, idx]
            
            # Mutation: randomly change some genes
            idx = np.random.rand(dim) < mutation_rate
            genes = x_min + np.random.rand(dim) * (x_max - x_min)
            offsprings[i, idx] = genes[idx]

        # Evaluate new solutions
        fitness_new = truss_obj_with_penalty(offsprings)

        # Update reference point (ideal point)
        z = np.minimum(z, np.min(fitness_new, axis=0))

        for i in range(n_subproblems):
            lambda_ = lambdas[i, :]
            g = tchebycheff(fitness[i, :], lambda_, z)
            
            for j in range(n_neighbors):
                neighbour = neighbourhood[i, j]
                h = tchebycheff(fitness_new[neighbour, :], lambda_, z)
                if h < g:
                    g = h
                    population[i, :] = offsprings[neighbour, :]
                    fitness[i, :] = fitness_new[neighbour, :]

        f_min = np.min(fitness, axis=0)
        print(f'gen: {gen}, {f_min[0]:.2f}, {f_min[1]:.2f}')
    return population, fitness

# Define the penalty method objective functions
def truss_obj_with_penalty(x):
    f1 = x[:, 0] * np.sqrt(16 + x[:, 2]**2) + x[:, 1] * np.sqrt(1 + x[:, 2]**2)
    f2 = 20 * np.sqrt(16 + x[:, 2]**2) / (x[:, 0] * x[:, 2])
    
    # Constraints
    g1 = f1 - 0.1
    g2 = f2 - 1e5
    g3 = 80 * np.sqrt(1 + x[:, 2]**2) / (x[:, 1] * x[:, 2]) - 1e5
    
    # Penalty parameters
    k11, k21, k31 = 250, 0.125, 2.5e-2
    k12, k22, k32 = 2.5e6, 1250, 250
    
    # Penalties
    penalty1 = k11 * np.maximum(g1, 0) + k21 * np.maximum(g2, 0) + k31 * np.maximum(g3, 0)
    penalty2 = k12 * np.maximum(g1, 0) + k22 * np.maximum(g2, 0) + k32 * np.maximum(g3, 0)
    
    # Total objectives with penalties
    h1 = f1 + penalty1
    h2 = f2 + penalty2
    # Scale objectives for normalization
    h1 *= 100      # Scale first objective for better visualization
    h2 /= 1000     # Scale second objective for better visualization
    return np.column_stack((h1, h2))

# Tchebycheff decomposition approach
def tchebycheff(fitness, lambda_, ideal):
    return np.max(np.abs(fitness - ideal) * lambda_)

# Function to check constraints
def check_constraints(x):
    """Check if the design variables satisfy all constraints."""
    f1 = x[0] * np.sqrt(16 + x[2]**2) + x[1] * np.sqrt(1 + x[2]**2)
    f2 = 20 * np.sqrt(16 + x[2]**2) / (x[0] * x[2])
    g3 = 80 * np.sqrt(1 + x[2]**2) / (x[1] * x[2])
    
    # Check constraints
    c1 = f1 <= 0.1
    c2 = f2 <= 1e5
    c3 = g3 <= 1e5
    
    return c1 and c2 and c3

# Algorithm parameters
max_gen = 500
num_subproblems = 400
neighbourhood_size = 20
mutation_rate = 0.2

# Run MOEA/D optimization
population, fitness = MOEAD(max_gen, num_subproblems, neighbourhood_size, mutation_rate)

# Plot the results (approximated Pareto front)
plt.scatter(fitness[:, 0], fitness[:, 1], marker='o')
plt.title('Approximated Pareto Front for Two-Bar Truss')
plt.xlabel('f1 * 100')
plt.ylabel('f2 / 1000')
plt.show()

# Verify constraints for each solution in the Pareto front
feasible = []
infeasible = []
for i in range(len(population)):
    if check_constraints(population[i]):
        feasible.append(i)
    else:
        infeasible.append(i)

print(f"\nNumber of feasible solutions: {len(feasible)}")
print(f"Number of infeasible solutions: {len(infeasible)}")

# Plot feasible and infeasible solutions separately
plt.figure()
if len(feasible) > 0:
    plt.scatter(fitness[feasible, 0], fitness[feasible, 1], c='b', marker='o', label='Feasible')
if len(infeasible) > 0:
    plt.scatter(fitness[infeasible, 0], fitness[infeasible, 1], c='r', marker='x', label='Infeasible')
plt.title('Pareto Front with Feasibility Check')
plt.xlabel('f1 * 100')
plt.ylabel('f2 / 1000')
plt.legend()
plt.show()