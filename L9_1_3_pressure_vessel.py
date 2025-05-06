import numpy as np
import pygad
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Define the objective function
def pres_ves_obj(x):
    """
    Objective function for the constrained pressure vessel design problem.
    """
    y1 = x[0] * x[2] * x[3]
    y2 = x[1] * x[2] * x[2]
    y3 = x[0] * x[0] * x[3]
    y4 = x[0] * x[0] * x[2]
    return 0.6224 * y1 + 1.7781 * y2 + 3.1661 * y3 + 19.84 * y4

# Define the constraint functions
def shell_thickness_radius_constraint1(x):
    return 0.0193 * x[2] - x[0]

def head_thickness_radius_constraint2(x):
    return 0.00954 * x[2] - x[1]

def capacity_constraint3(x):
    return 1296000.0 - np.pi * x[2]**2 * x[3] - (4 / 3) * np.pi * x[2]**3

def length_constraint4(x):
    return x[3] - 240

# Fitness function for constrained optimization using pygad
def fitness_func(ga_instance, solution, solution_idx):
    """
    Calculates fitness for pressure vessel design problem with constraints.
    Implements penalty function method for constraint handling.  
    Args:
        ga_instance: pygad.GA instance
        solution: Current solution vector
        solution_idx: Index of solution in population        
    Returns:
        float: Fitness value (negative for minimization)
    """
    x = solution.copy()  # Copy solution to avoid modifying original solution
    x[0] = 0.0625 * x[0]  # Shell thickness conversion
    x[1] = 0.0625 * x[1]  # Head thickness conversion
    
    # Evaluate constraint violations
    cv1 = shell_thickness_radius_constraint1(x)  # Shell thickness-radius ratio
    cv2 = head_thickness_radius_constraint2(x)  # Head thickness-radius ratio
    cv3 = capacity_constraint3(x)               # Minimum capacity requirement
    cv4 = length_constraint4(x)                 # Total length limitation
    
    # Penalty coefficients (pre-tuned values)
    k1 = 18750.0  # Weight for shell thickness violation
    k2 = 37500.0  # Weight for head thickness violation
    k3 = 750.0    # Weight for capacity violation
    k4 = 7.5e6    # Weight for length violation
    
    # Calculate base objective (pressure vessel weight)
    obj = pres_ves_obj(x)
    
    # Apply quadratic penalty for constraint violations
    penalty = (k1 * max(cv1, 0) + k2 * max(cv2, 0) + k3 * max(cv3, 0) + k4 * max(cv4, 0))
    
    # Combine objective and penalties
    fitness = obj + penalty
    
    # Convert to negative for minimization in pygad
    return -fitness

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

    # Convert population to a homogeneous array of floats
    population = population.astype(float)

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

# Genetic Algorithm Configuration Parameters
max_generations = 300    # Maximum number of evolutionary generations
sol_per_pop = 50         # Number of solutions (chromosomes) in each population
num_genes = 4            # Number of design variables in the optimization problem

# Define search space for each design variable
gene_space = [
    # Discrete variables (integer values)
    {'low': 1, 'high': 99},    # x1: Shell thickness (discrete inch values)
    {'low': 1, 'high': 99},    # x2: Head thickness (discrete inch values)
    
    # Continuous variables
    {'low': 10.0, 'high': 200.0},  # x3: Inner radius (continuous in cm)
    {'low': 10.0, 'high': 200.0}   # x4: Vessel length (continuous in cm)
]

# Initialize Genetic Algorithm instance
ga_instance = pygad.GA(
    # Core evolutionary parameters
    num_generations=max_generations,  # Termination criterion
    num_parents_mating=sol_per_pop,   # Number of solutions selected for reproduction
    sol_per_pop=sol_per_pop,          # Population size maintained each generation
    
    # Problem-specific configuration
    fitness_func=fitness_func,        # Custom fitness evaluation function
    num_genes=num_genes,              # Matches number of design variables
    gene_space=gene_space,            # Defined search space boundaries
    
    # Genetic operators configuration
    mutation_type="random",           # Random value replacement mutation
    parent_selection_type="tournament",  # Competitive selection mechanism
    crossover_type="single_point",    # Genetic recombination method
    crossover_probability=0.8,        # Probability of crossover occurring
    
    # Mutation parameters
    mutation_num_genes=2,             # Maximum genes to mutate per solution
    
    # Variable type specification
    gene_type=[int, int, float, float],  # Data types for x1-x4 respectively
    
    # Optional callback function
    on_generation=on_generation       # Called after each generation
)

# Run the genetic algorithm
ga_instance.run()

# Output results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution (selected items): ", solution)
print("Fitness value of the best solution: ", -solution_fitness)

# Keep the plot window open
plt.ioff()
plt.show()