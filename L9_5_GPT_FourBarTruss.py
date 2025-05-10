import numpy as np
import matplotlib.pyplot as plt

# Define the new objective functions based on the provided equations
F = 10  # kN
sigma = 10  # kN/cm^2
E = 2 * 10**5  # kN/(cm^2)
L = 200  # cm
c = 10**-4

# Define the new problem
def mop_objective_function(x):
    f1 = c * L * (2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(x[2]) + x[3])
    f2 = (F * L) / E * (2 / x[0] + 2 * np.sqrt(2) / x[1] - 2 * np.sqrt(2) / x[2] + 2 / x[3])
    return np.array([f1, f2])

# Generate weight vectors
def generate_weight_vectors(n_vectors, n_dimensions):
    vectors = np.random.dirichlet(np.ones(n_dimensions), size=n_vectors)
    return vectors

# Tchebycheff decomposition
def tchebycheff(zi, z, weight_vector):
    return np.max(weight_vector * np.abs(zi - z))

# Initialize population within constraints
def initialize_population(pop_size, n_dimensions):
    x = np.zeros((pop_size, n_dimensions))
    for i in range(pop_size):
        x[i, 0] = np.random.uniform(F / sigma, 3 * F / sigma)
        x[i, 1] = np.random.uniform(np.sqrt(2) * F / sigma, 3 * F / sigma)
        x[i, 2] = np.random.uniform(np.sqrt(2) * F / sigma, 3 * F / sigma)
        x[i, 3] = np.random.uniform(F / sigma, 3 * F / sigma)
    return x

# Evaluate population
def evaluate_population(population):
    return np.array([mop_objective_function(ind) for ind in population])

# Genetic algorithm operators
def tournament_selection(population, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for p in range(num_parents):
        i, j = np.random.choice(np.arange(len(population)), size=2, replace=False)
        if np.random.rand() > 0.5:
            parents[p] = population[i]
        else:
            parents[p] = population[j]
    return parents

def crossover(parents, crossover_rate):
    offspring = np.empty_like(parents)
    for k in range(0, len(parents), 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, parents.shape[1] - 1)
            offspring[k] = np.hstack((parents[k, 0:crossover_point], parents[k + 1, crossover_point:]))
            offspring[k + 1] = np.hstack((parents[k + 1, 0:crossover_point], parents[k, crossover_point:]))
        else:
            offspring[k], offspring[k + 1] = parents[k], parents[k + 1]
    return offspring

def mutation(offspring, mutation_rate):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[i, j] = np.random.uniform(
                    F / sigma if j == 0 or j == 3 else np.sqrt(2) * F / sigma, 
                    3 * F / sigma
                )
    return offspring

# MOEA/D
def moea_d(pop_size, n_dimensions, n_generations):
    # Parameters
    num_parents = pop_size
    crossover_rate = 0.9
    mutation_rate = 0.1
    
    # Initialize population and evaluate
    population = initialize_population(pop_size, n_dimensions)
    objectives = evaluate_population(population)
    
    # Generate weight vectors
    weight_vectors = generate_weight_vectors(pop_size, 2)
    
    # Ideal point
    z = np.min(objectives, axis=0)
    
    # Main loop
    for gen in range(n_generations):
        # Selection and reproduction
        parents = tournament_selection(population, num_parents)
        offspring = crossover(parents, crossover_rate)
        offspring = mutation(offspring, mutation_rate)
        
        # Evaluate offspring
        offspring_objectives = evaluate_population(offspring)
        
        # Update population
        for k in range(pop_size):
            # Decomposition method is Tchebycheff
            for o_ind, o_obj in zip(offspring, offspring_objectives):
                if tchebycheff(o_obj, z, weight_vectors[k]) < tchebycheff(objectives[k], z, weight_vectors[k]):
                    population[k] = o_ind
                    objectives[k] = o_obj

        # Update ideal point
        z = np.minimum(z, np.min(offspring_objectives, axis=0))
        print(f'gen: {gen}, {z[0]:.2f}, {z[1]:.2f}')
    
    return population, objectives

# Parameters
pop_size = 150
n_dimensions = 4
n_generations = 200

# Run MOEA/D
population, objectives = moea_d(pop_size, n_dimensions, n_generations)

# Plot results
plt.scatter(objectives[:, 0], objectives[:, 1], c='b', marker='o')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Pareto Front for MOP')
plt.show()