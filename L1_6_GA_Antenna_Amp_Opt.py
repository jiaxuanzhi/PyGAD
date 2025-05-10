import numpy as np
import pygad
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.constants import speed_of_light

# Define the PSLL Objective Function
def objective_psll(I0):
    # Basic Electromagnetic Parameters
    Frequency = 10e9
    Lightspeed = speed_of_light
    Wavelength = Lightspeed / Frequency
    Wavenumber = 2 * np.pi / Wavelength

    # Array Parameters
    N = len(I0)
    X = np.arange(1, N + 1) * Wavelength * 0.5
    alpha = np.zeros(N)

    # ArrayFactor Sampling
    Ns = 1001  # Sampling number
    theta = np.linspace(-90, 90, Ns)

    # Uniform Array
    E0 = np.zeros(Ns, dtype=complex)
    for num in range(Ns):
        E0[num] = np.sum(I0 * np.exp(1j * (Wavenumber * X * np.sin(np.deg2rad(theta[num])) + alpha)))

    E0_dB = 20 * np.log10(np.abs(E0)) - np.max(20 * np.log10(np.abs(E0)))  # normalized pattern

    # Calculate PSLL
    MPeak = np.argmax(E0_dB)
    AF_inc = np.diff(E0_dB[MPeak:]) > 0  # precompute the pattern diff
    SLL_start = np.where(AF_inc)[0][0] if np.any(AF_inc) else len(E0_dB) - MPeak  # find the first null
    SLL = np.max(E0_dB[MPeak + SLL_start-1:])

    return SLL


import numpy as np
from scipy.constants import speed_of_light
import matplotlib.pyplot as plt

def PlotPattern(N, I0):
    # Basic Electromagnetic Parameters
    Frequency = 10e9
    Lightspeed = speed_of_light
    Wavelength = Lightspeed / Frequency
    Wavenumber = 2 * np.pi / Wavelength

    # Array Parameters
    X = np.arange(1, N + 1) * Wavelength * 0.5
    alpha = np.zeros(N)

    # ArrayFactor Sampling
    Ns = 1001  # Sampling number
    theta = np.linspace(-90, 90, Ns)

    # Uniform Array
    E0 = np.zeros(Ns, dtype=complex)
    for num in range(Ns):
        E0[num] = np.sum(I0 * np.exp(1j * (Wavenumber * X * np.sin(np.deg2rad(theta[num])) + alpha)))

    E0_dB = 20 * np.log10(np.abs(E0)) - np.max(20 * np.log10(np.abs(E0)))  # normalized pattern

    # Plot figure
    plt.figure()
    plt.plot(theta, E0_dB, linewidth=2)
    # plt.ylim([-50, 0])
    plt.grid(True)
    plt.xlabel(r'$\theta$ (degrees)')
    plt.ylabel('dB')
    plt.tight_layout()

    plt.figure()
    plt.scatter(np.arange(1, N + 1), I0)
    plt.show()
    
# Define fitness function (updated to accept 3 parameters)
def fitness_func(ga_instance, solution, solution_idx):
    # Use the objective_psll function to calculate the fitness
    return -objective_psll(solution)  # Minimization problem, return negative value

# Initialize lists to store statistics
mean_fitness_history = []  # Store mean fitness of each generation
std_fitness_history = []   # Store standard deviation of fitness of each generation
diversity_history = []     # Store population position standard deviation (diversity) of each generation
best_fitness_history = []  # Store best fitness value of each generation
# maxGeneValue = 500
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
# ax1.set_xlim(0, maxGeneValue)
# ax1.set_ylim(0, 100)

line2, = ax2.plot([], [], '.b', label="Fitness Std")
ax2.set_xlabel("Generation")
ax2.set_ylabel("Standard deviation")
ax2.set_title("Fitness Standard Deviation: N/A")
ax2.legend()
# ax2.set_xlim(0, maxGeneValue)
# ax2.set_ylim(0, 20)

line3, = ax3.plot([], [], '.g', label="Position Std")
ax3.set_xlabel("Generation")
ax3.set_ylabel("Standard deviation")
ax3.set_title("Position Standard Deviation: N/A")
ax3.legend()
# ax3.set_xlim(0, maxGeneValue)
# ax3.set_ylim(0, 20)

plt.tight_layout()
# Add buttons for pause and continue
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

    # Plot the pattern of the best solution
    if ga_instance.generations_completed % maxGeneValue == 0:  # Plot every 20 generations
        best_solution, best_fitness, _ = ga_instance.best_solution()
        PlotPattern(len(best_solution), best_solution)

    if ga_instance.generations_completed % maxGeneValue == 0:  # Plot end of the generation of the original pattern       
        PlotPattern(len(best_solution), np.ones(len(best_solution)))
# Define variable range
num_genes = 11  # Dimension of the problem
maxGeneValue = 150 # Number of generations
sol_per_pop = 20  # Population size
gene_space = [{'low': 0, 'high': 1} for _ in range(num_genes)]
# initial_population = np.random.uniform(low=1, high=10, size=(sol_per_pop, num_genes))
# Create genetic algorithm instance
ga_instance = pygad.GA(
    num_generations = maxGeneValue,  # Number of iterations
    # stop_criteria="saturate_500",  
    # Stop criteria, stop if fitness value does not change for 500 generations
    num_parents_mating = sol_per_pop,  # Number of parents for mating
    sol_per_pop = sol_per_pop,  # Population size
    fitness_func = fitness_func,  # Fitness function
    num_genes = num_genes,  # Dimension of the problem
    gene_space = gene_space,  # Variable range
    # initial_population = initial_population,  # Set initial population
    mutation_type="random", # random, adaptive
    parent_selection_type = "tournament", 
    # include: rws (Roulette Wheel Selection), sss (Steady-State Selection),
    # sus (Stochastic Universal Sampling),rank (Rank Selection),
    # random (Random Selection),tournament (Tournament Selection).
    keep_parents = 1,  # Number of parents to keep in the next generation.
    keep_elitism = 1, # Number of elites to keep in the next generation, default is 1.
    crossover_type = "single_point", # single_point, two_points, uniform, scattered.
    crossover_probability = 0.8, # Crossover probability
    mutation_probability = 0.01,  # Mutation probability
    # mutation_percent_genes = 0.1,  # Mutation probability
    # random_seed = 2, # Random seed for reproducibility
    parallel_processing = 5, # Number of parallel processes to use, default is 1.
    gene_type = float, # Gene type,  [float, 3]  means 3 floating point numbers
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
# plt.plot(np.linspace(101, 200, 100),best_fitness_history[100:200], label="Fitness Std", color="orange")
# plt.tight_layout()
# plt.show()