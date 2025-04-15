import numpy as np
import pygad
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define the problem parameters
values = np.array([10, 15, 30, 25, 40])  # Values of items
weights = np.array([1, 2, 2, 3, 4])      # Weights of items
volumes = np.array([10, 20, 25, 15, 30]) # Volumes of items
max_weight = 7                            # Maximum weight allowed
max_volume = 60                           # Maximum volume allowed
w_penalty = 5000                          # Penalty coefficient for weight constraint
v_penalty = 1000                          # Penalty coefficient for volume constraint

# Define the fitness function for the knapsack problem
def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness function for the knapsack problem.
    Uses the penalty function method to handle constraints.
    """
    # Calculate the total value of the selected items
    total_value = np.dot(values, solution)
    # Calculate the total weight and volume of the selected items
    total_weight = np.dot(weights, solution)
    total_volume = np.dot(volumes, solution)
    # Apply penalty for weight and volume constraints
    weight_penalty = w_penalty * max(0, total_weight - max_weight)
    volume_penalty = v_penalty * max(0, total_volume - max_volume)
    # Fitness is the total value minus penalties (convert to minimization problem)
    fitness = total_value - weight_penalty - volume_penalty
    # Since pygad maximizes fitness, return the fitness directly
    return fitness

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

plt.tight_layout()

# Add buttons for pause and continue
ax_pause = plt.axes([0.05, 0.93, 0.1, 0.05])  # Pause button position
ax_continue = plt.axes([0.17, 0.93, 0.1, 0.05])  # Continue button position 
btn_pause = Button(ax_pause, 'Pause')
btn_continue = Button(ax_continue, 'Continue')

# Global variable to control pause state
paused = False

# Define callback functions for buttons
def pause(event):
    global paused
    paused = True

def continue_execution(event):
    global paused
    paused = False

# Attach button callbacks
btn_pause.on_clicked(pause)
btn_continue.on_clicked(continue_execution)

# Define callback function to be called at the end of each generation
def on_generation(ga_instance):
    global paused

    # Check if paused
    while paused:
        plt.pause(0.1)  # Pause for a short time to avoid busy-waiting

    # Get current population
    population = ga_instance.population

    # Calculate current generation's fitness values
    fitness_values = ga_instance.last_generation_fitness

    # Save the current generation's fitness best values
    best_fitness = np.max(fitness_values)
    best_fitness_history.append(best_fitness)
    
    # Calculate mean and standard deviation of fitness
    mean_fitness = np.mean(fitness_values)
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
    ax1.set_title(f"Best Fitness: {best_fitness:.2f}")
    ax1.relim()
    ax1.autoscale_view()

    line2.set_xdata(np.arange(len(std_fitness_history)))
    line2.set_ydata(std_fitness_history)
    ax2.set_title(f"Fitness Standard Deviation: {std_fitness:.2f}")
    ax2.relim()
    ax2.autoscale_view()

    line3.set_xdata(np.arange(len(diversity_history)))
    line3.set_ydata(diversity_history)
    ax3.set_title(f"Position Standard Deviation: {diversity:.2f}")
    ax3.relim()
    ax3.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

# Define variable range
num_genes = len(values)  # Total number of parameters (genes) to optimize
max_generations = 100    # Maximum iterations for the GA algorithm
sol_per_pop = 50         # Number of solutions (chromosomes) per population

# Define search space for each gene (parameter bounds)
gene_space = [{'low': -0.1, 'high': 1.1}] * num_genes  

# Create genetic algorithm instance
ga_instance = pygad.GA(
    num_generations=max_generations,  # Stopping criterion
    num_parents_mating=sol_per_pop,   # Number of parents for mating
    sol_per_pop=sol_per_pop,          # Population size
    fitness_func=fitness_func,        # Custom fitness evaluation function
    num_genes=num_genes,              # Matches the problem dimension
    gene_space=gene_space,            # Defined parameter bounds
    mutation_type="random",           # Random value mutation
    parent_selection_type="tournament",  # Selection method
    crossover_type="single_point",    # Genetic recombination method
    crossover_probability=0.8,        # Chance of crossover occurring
    mutation_percent_genes=1,         # Percentage of genes to mutate
    gene_type=int,                    # Force integer-valued solutions
    on_generation=on_generation       # Optional generation callback
)

# Run the genetic algorithm
ga_instance.run()

# Output results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution (selected items): ", solution)
print("Fitness value of the best solution: ", solution_fitness)

# Keep the plot window open
plt.ioff()
plt.show()