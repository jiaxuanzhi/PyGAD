import numpy as np
import pygad
import matplotlib.pyplot as plt

# Define Rastrigin function
def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Define fitness function (updated to accept 3 parameters)
def fitness_func(ga_instance, solution, solution_idx):
    return -rastrigin_function(solution)  # Minimization problem, return negative value

# Initialize lists to store statistics
mean_fitness_history = []  # Store mean fitness of each generation
std_fitness_history = []   # Store standard deviation of fitness of each generation
diversity_history = []     # Store population position standard deviation (diversity) of each generation
best_fitness_history = []  # Store best fitness value of each generation

# Define callback function to be called at the end of each generation
def on_generation(ga_instance):
    # Get current population
    population = ga_instance.population

    # Calculate current generation's fitness values
    fitness_values = ga_instance.last_generation_fitness

    # save the current generation's fitness best values
    best_fitness = -np.max(fitness_values)
    best_fitness_history.append(best_fitness)
    
    # Calculate mean and standard deviation of fitness
    mean_fitness = -np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    mean_fitness_history.append(mean_fitness)
    std_fitness_history.append(std_fitness)

    # Calculate population average point
    population_average = np.mean(population, axis=0)

    # Calculate diversity (Population position standard deviation)
    diversity = np.mean(np.sqrt(np.sum((population - population_average) ** 2, axis=1)))
    diversity_history.append(diversity)

# Define variable range
num_genes = 10  # Dimension of the problem
maxGeneValue = 400 # Number of generations
sol_per_pop = 20  # Population size
gene_space = [{'low': -10.12, 'high': 10.12} for _ in range(num_genes)]
# Create genetic algorithm instance
ga_instance = pygad.GA(
    num_generations = maxGeneValue,  # Number of iterations
    stop_criteria="saturate_500",  # Stop criteria, stop if fitness value does not change for 500 generations
    num_parents_mating = sol_per_pop,  # Number of parents for mating
    sol_per_pop = sol_per_pop,  # Population size
    fitness_func = fitness_func,  # Fitness function
    num_genes = num_genes,  # Dimension of the problem
    gene_space = gene_space,  # Variable range
    mutation_type="random", # random, adaptive
    parent_selection_type = "tournament", # include: rws (Roulette Wheel Selection), sss (Steady-State Selection),
    # sus (Stochastic Universal Sampling),rank (Rank Selection),random (Random Selection),tournament (Tournament Selection)
    keep_parents = 1,  # Number of parents to keep in the next generation
    crossover_type = "single_point", # single_point, two_points, uniform, scattered
    crossover_probability = 0.8, # Crossover probability
    mutation_percent_genes = 0.1,  # Mutation probability
    on_generation = on_generation  # Callback function at the end of each generation
)

# Run the genetic algorithm
ga_instance.run()

# Output results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution: ", solution)
print("Fitness value of the best solution: ", -solution_fitness)  # Negate to get Rastrigin function value

# Plot fitness over generations
# ga_instance.plot_fitness()

# Plot mean fitness, standard deviation of fitness, and diversity
plt.figure(figsize=(9, 6))

# Plot mean fitness
plt.subplot(3, 1, 1)
plt.plot(best_fitness_history, label="Current best fitness", color="black")  #  mean_fitness_history
plt.xlabel("Generation")
plt.ylabel("Fitness value")
plt.title("Best: " + str(-solution_fitness))
# plt.ylim(0, 100)
plt.xlim(0, maxGeneValue)
plt.legend()

# Plot standard deviation of fitness
plt.subplot(3, 1, 2)
plt.plot(std_fitness_history, label="Fitness Std", color="orange")
plt.xlabel("Generation")
plt.ylabel("Standard deviation")
plt.title("Fitness Std: " + str(std_fitness_history[-1]))
# plt.ylim(0, 100)
plt.xlim(0, maxGeneValue)
plt.legend()

# Plot diversity (Population position standard deviation)
plt.subplot(3, 1, 3)
plt.plot(diversity_history, label="Position Std", color="green")
plt.xlabel("Generation")
plt.ylabel("Standard deviation")
plt.title("Position Std: " + str(diversity_history[-1]))
# plt.ylim(0, 20)
plt.xlim(0, maxGeneValue)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
plt.plot(np.linspace(101, 200, 100),best_fitness_history[100:200], '.b',label="Fitness Std", color="orange")
plt.tight_layout()
plt.show()