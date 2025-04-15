import numpy as np
import pygad
import matplotlib.pyplot as plt

# # Define Ackley function
# def ackley_function(x):
#     a = 20
#     b = 0.2
#     c = 2 * np.pi
#     n = len(x)
#     sum_sq_term = -a * np.exp(-b * np.sqrt(sum([xi**2 for xi in x]) / n))
#     cos_term = -np.exp(sum([np.cos(c * xi) for xi in x]) / n)
#     return sum_sq_term + cos_term + a + np.exp(1)

# # Define fitness function (updated to accept 3 parameters)
# def fitness_func(ga_instance, solution, solution_idx):
#     return -ackley_function(solution)  # Minimization problem, return negative value

# Define Rastrigin function
def sphere_function(x):
    n = len(x)
    return sum([(xi**2) for xi in x])

# Define fitness function (updated to accept 3 parameters)
def fitness_func(ga_instance, solution, solution_idx):
    return -sphere_function(solution)  # Minimization problem, return negative value

# Initialize lists to store statistics
mean_fitness_history = []  # Store mean fitness of each generation
std_fitness_history = []   # Store standard deviation of fitness of each generation
diversity_history = []     # Store population position standard deviation (diversity) of each generation
best_fitness_history = []  # Store best fitness value of each generation

# Initialize plot
plt.ion()  # Enable interactive mode
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
fig.suptitle('Genetic Algorithm Statistics')

# Plot initialization
line1, = ax1.plot([], [], '.k', label="Current best fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness value")
ax1.set_title("Best Fitness: N/A")
ax1.legend()

line2, = ax2.plot([], [], '.b', label="Fitness Std")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Standard deviation")
ax2.set_title("Fitness Standard Deviation: N/A")
ax2.legend()

line3, = ax3.plot([], [], '.g', label="Position Std")
ax3.set_xlabel("Generation")
ax3.set_ylabel("Standard deviation")
ax3.set_title("Position Standard Deviation: N/A")
ax3.legend()
# ax3.set_xlim(0, 400)
# ax3.set_ylim(0, 20)

plt.tight_layout()
# Define callback function to be called at the end of each generation
def on_generation(ga_instance):
    # Get current population
    population = ga_instance.population

    # Calculate current generation's fitness values
    fitness_values = ga_instance.last_generation_fitness

    # Save the current generation's fitness best values
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

    # Update plots
    line1.set_xdata(np.arange(len(best_fitness_history)))
    line1.set_ydata(best_fitness_history)
    ax1.set_title(f"Best Fitness: {best_fitness:.8f}")
    ax1.relim()
    ax1.autoscale_view()

    line2.set_xdata(np.arange(len(std_fitness_history)))
    line2.set_ydata(std_fitness_history)
    ax2.set_title(f"Fitness Standard Deviation: {std_fitness:.8f}")
    ax2.relim()
    ax2.autoscale_view()

    line3.set_xdata(np.arange(len(diversity_history)))
    line3.set_ydata(diversity_history)
    ax3.set_title(f"Position Standard Deviation: {diversity:.8f}")
    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

# Define variable range
num_genes = 20  # Dimension of the problem
maxGeneValue = 100 # Number of generations
sol_per_pop = 200  # Population size
gene_space = [{'low': -10.12, 'high': 10.12} for _ in range(num_genes)]
# Create genetic algorithm instance
ga_instance = pygad.GA(
    num_generations = maxGeneValue,  # Number of iterations
    stop_criteria="saturate_500",  
    # Stop criteria, stop if fitness value does not change for 500 generations
    num_parents_mating = sol_per_pop,  # Number of parents for mating
    sol_per_pop = sol_per_pop,  # Population size
    fitness_func = fitness_func,  # Fitness function
    num_genes = num_genes,  # Dimension of the problem
    gene_space = gene_space,  # Variable range
    mutation_type="random", # random, adaptive
    # mutation_type=None, # random, adaptive
    parent_selection_type = "tournament", 
    # e.g.:  rws (Roulette Wheel Selection), sss (Steady-State Selection),
    #           sus (Stochastic Universal Sampling),rank (Rank Selection),
    #           random (Random Selection),tournament (Tournament Selection).
    keep_parents = 1,  # Number of parents to keep in the next generation
    crossover_type = "single_point", # single_point, two_points, uniform, scattered
    crossover_probability = 0.8, # Crossover probability
    mutation_percent_genes = 1,  # Mutation probability
    on_generation = on_generation  # Callback function at the end of each generation
)

# Run the genetic algorithm
ga_instance.run()

# Output results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution: ", solution)
print("Fitness value of the best solution: ", -solution_fitness)  # Negate to get Rastrigin function value

# Keep the plot window open
plt.ioff()
plt.show()

# plt.figure(figsize=(9, 6))
# plt.scatter(range(len(best_fitness_history)), best_fitness_history, label="Fitness Std", color="black")
# plt.title(f"Fitness Standard Deviation: {std_fitness:.8f}")
# plt.tight_layout()
# plt.show()


# 保存第一次运行的 best_fitness_history
np.save('best_fitness_history_run4.npy', best_fitness_history)

# 加载第一次运行的 best_fitness_history
best_fitness_history_run1 = np.load('best_fitness_history_run3.npy')


# 绘制两次运行的结果
plt.scatter(range(len(best_fitness_history_run1)), best_fitness_history_run1, label="crossover probability = 0.1", color="blue", marker='o',s=10)
plt.scatter(range(len(best_fitness_history)), best_fitness_history, label="crossover probability = 0.8", color="red", marker='o',s=10)

# 添加标题和标签
plt.title("Best Fitness History Comparison")
plt.xlabel("Iteration")
plt.ylabel("Fitness Value")

# 添加图例
plt.legend()

# 显示图形
plt.show()

# 保存图形
# plt.savefig('fitness_comparison.png')