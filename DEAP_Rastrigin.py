import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Define Rastrigin function
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Define fitness function
def evaluate(individual):
    return rastrigin_function(individual),  # DEAP requires a tuple

# Define variable range
num_genes = 2  # Dimension of the problem
lower_bound = -10.12
upper_bound = 10.12

# Initialize DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.uniform, lower_bound, upper_bound)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # Gaussian mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

# Initialize lists to store statistics
mean_fitness_history = []  # Store mean fitness of each generation
std_fitness_history = []   # Store standard deviation of fitness of each generation
diversity_history = []     # Store population position standard deviation (diversity) of each generation
best_fitness_history = []  # Store best fitness value of each generation

# Define a function to calculate statistics
def calculate_stats(population):
    fitness_values = [ind.fitness.values[0] for ind in population]
    mean_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    best_fitness = np.min(fitness_values)
    
    # Calculate diversity (Population position standard deviation)
    population_array = np.array(population)
    population_average = np.mean(population_array, axis=0)
    diversity = np.mean(np.sqrt(np.sum((population_array - population_average) ** 2, axis=1)))
    
    return mean_fitness, std_fitness, best_fitness, diversity

# Define a callback function to log statistics
def log_stats(population, generation):
    mean_fitness, std_fitness, best_fitness, diversity = calculate_stats(population)
    mean_fitness_history.append(mean_fitness)
    std_fitness_history.append(std_fitness)
    best_fitness_history.append(best_fitness)
    diversity_history.append(diversity)
    print(f"Generation {generation}: Mean Fitness = {mean_fitness}, Best Fitness = {best_fitness}")

# Create population
population = toolbox.population(n=10)

# Evaluate the entire population
fitness_values = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitness_values):
    ind.fitness.values = fit

# Run the genetic algorithm
num_generations = 500
for generation in range(num_generations):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.8:  # Crossover probability
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    for mutant in offspring:
        if random.random() < 0.2:  # Mutation probability
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate the individuals with invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitness_values = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness_values):
        ind.fitness.values = fit
    
    # Replace the population with the offspring
    population[:] = offspring
    
    # Log statistics
    log_stats(population, generation)

# Output results
best_individual = tools.selBest(population, k=1)[0]
print("Best solution: ", best_individual)
print("Fitness value of the best solution: ", best_individual.fitness.values[0])

# Plot mean fitness, standard deviation of fitness, and diversity
plt.figure(figsize=(7, 6))

# Plot mean fitness
plt.subplot(3, 1, 1)
plt.plot(best_fitness_history, label="Current best fitness", color="black")
plt.xlabel("Generation")
plt.ylabel("Fitness value")
plt.title("Best: " + str(best_individual.fitness.values[0]))
plt.legend()

# Plot standard deviation of fitness
plt.subplot(3, 1, 2)
plt.plot(std_fitness_history, label="Fitness Std", color="orange")
plt.xlabel("Generation")
plt.ylabel("Standard deviation")
plt.title("Fitness Std: " + str(std_fitness_history[-1]))
plt.legend()

# Plot diversity (Population position standard deviation)
plt.subplot(3, 1, 3)
plt.plot(diversity_history, label="Position Std", color="green")
plt.xlabel("Generation")
plt.ylabel("Standard deviation")
plt.title("Position Std: " + str(diversity_history[-1]))
plt.legend()

plt.tight_layout()
plt.show()