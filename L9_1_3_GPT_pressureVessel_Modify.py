import pygad
import numpy as np

# Constants given in the problem
a1, a2, a3, a4 = 0.6224, 1.7781, 3.1611, 19.84
c1, c2, c3, c4 = 0.0193, 0.00954, 1296000, 240
k1, k2, k3, k4 = 18750, 37500, 750, 7.5e6  # Penalty coefficients

# Objective function integrating constraints as penalties
def fitness_func(GA_instantce,solution, solution_idx):
    b1, b2, x3, x4 = solution
    x1 = 0.0625 * b1  # Shell thickness in inches
    x2 = 0.0625 * b2  # Head thickness in inches
    penalty = (k1 * max(c1 * x3 - x1, 0) +
               k2 * max(c2 * x3 - x2, 0) +
               k3 * max(c3 - np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3, 0) +
               k4 * max(x4 - c4, 0))
    
    objective_value = (a1 * x1 * x3 * x4 +
                       a2 * x2 * x3**2 +
                       a3 * x1**2 * x4 +
                       a4 * x1**2 * x3 +
                       penalty)
    return -objective_value  # Maximize negative of the objective for minimization

# Genetic Algorithm parameters
num_generations = 400
sol_per_pop = 50
num_parents_mating = sol_per_pop
num_genes = 4
parent_selection_type = "tournament"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

# x1 and x2 are discrete variables
gene_space = [    
    {'low': 1, 'high': 99},    # x1: Shell thickness (discrete inch values)
    {'low': 1, 'high': 99},    # x2: Head thickness (discrete inch values)
              {'low': 10, 'high': 200},  # x3
              {'low': 10, 'high': 200}]  # x4

# Initialize GA
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                    #    mutation_percent_genes=mutation_percent_genes,
                       mutation_num_genes=2,             # Maximum genes to mutate per solution
                       gene_type=[int, int, float, float]  # Data types for x1-x4 respectively
                       )

# Run GA
ga_instance.run()

# Output the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: x1={solution[0]}, x2={solution[1]}, x3={solution[2]}, x4={solution[3]}")
print(f"Objective function value: {-solution_fitness}")

# Plot result
ga_instance.plot_fitness(plot_type="scatter", title="Fitness over Generations")