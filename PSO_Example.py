import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyswarms as ps

# Define Rastrigin function
def rastrigin_function(x):
    A = 10
    n = x.shape[1]  # Number of dimensions
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)

# Initialize lists to store statistics
best_fitness_history = []  # Store best fitness value of each iteration
mean_fitness_history = []  # Store mean fitness of each iteration
std_fitness_history = []   # Store standard deviation of fitness of each iteration
diversity_history = []     # Store population position standard deviation (diversity) of each iteration

# Initialize plot
plt.ion()  # Enable interactive mode
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7))
fig.suptitle('Particle Swarm Optimization Statistics')

# Plot initialization
line1, = ax1.plot([], [], '.k', label="Current best fitness")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Fitness value")
ax1.set_title("Best Fitness: N/A")
ax1.legend()

line2, = ax2.plot([], [], '.b', label="Fitness Std")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Standard deviation")
ax2.set_title("Fitness Standard Deviation: N/A")
ax2.legend()

line3, = ax3.plot([], [], '.g', label="Position Std")
ax3.set_xlabel("Iteration")
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

# Define PSO callback function to be called at the end of each iteration
def pso_callback(swarm, fitness):
    global paused

    # Check if paused
    while paused:
        plt.pause(0.1)  # Pause for a short time to avoid busy-waiting

    # Calculate current iteration's fitness values
    fitness_values = fitness  # Fitness values of the swarm

    # Save the current iteration's fitness best values
    best_fitness = np.min(fitness_values)
    best_fitness_history.append(best_fitness)

    # Calculate mean and standard deviation of fitness
    mean_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)
    mean_fitness_history.append(mean_fitness)
    std_fitness_history.append(std_fitness)

    # Calculate population average point
    population_average = np.mean(swarm, axis=0)

    # Calculate diversity (Population position standard deviation)
    diversity = np.mean(np.sqrt(np.sum((swarm - population_average) ** 2, axis=1)))
    diversity_history.append(diversity)

    # Update plots
    line1.set_xdata(np.arange(len(best_fitness_history)))
    line1.set_ydata(best_fitness_history)
    ax1.set_title(f"Best Fitness: {best_fitness:f}")
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
num_dimensions = 5  # Dimension of the problem
bounds = (np.array([-10] * num_dimensions), np.array([10] * num_dimensions))  
# Lower and upper bounds for each dimension
options = {'c1': 1, 'c2': 1, 'w': 0.9}  # PSO parameters with GlobalBestPSO
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2} # PSO parameters with LocalBestPSO
swarm_size = 200  # Number of particles in the swarm
max_iterations = 1000  # Maximum number of iterations
# init_pos = np.random.uniform(0.1, 1.1, (swarm_size, num_dimensions))  
# Initial positions of the particles

# Initialize PSO optimizer with LocalBestPSO
# optimizer = ps.single.LocalBestPSO(
#     n_particles = swarm_size,
#     dimensions = num_dimensions,
#     options = options,
#     bounds = bounds,
#     # init_pos = init_pos  # Leave initialization to the optimizer
# )

# Initialize PSO optimizer with GlobalBestPSO
optimizer = ps.single.GlobalBestPSO(
    n_particles = swarm_size,
    dimensions = num_dimensions,
    options = options,
    bounds = bounds,
    # init_pos = init_pos  # Leave initialization to the optimizer
)

# Manually run iterations to support callback
for i in range(max_iterations):
    # Perform one iteration of PSO
    optimizer.optimize(rastrigin_function, iters=1)

    # Get current swarm and fitness values
    swarm = optimizer.swarm.position
    fitness = optimizer.swarm.current_cost

    # Call the callback function
    pso_callback(swarm, fitness)

# Output results
best_position = optimizer.swarm.best_pos
best_fitness = optimizer.swarm.best_cost
print("Best solution: ", best_position)
print("Fitness value of the best solution: ", best_fitness)

# Keep the plot window open
plt.ioff()
plt.show()