import numpy as np
import matplotlib.pyplot as plt

def GA_01(population_size, num_generations, mutation_rate, length, calculate_fitness):
    # Display the settings
    print('### Genetic Algorithm Settings:')
    print(f'    Population Size: {population_size}')
    print(f'    Number of Generations: {num_generations}')
    print(f'    Mutation Rate: {mutation_rate}')
    print('### Genetic Algorithm Start ...')

    n = length
    population = np.zeros((population_size, n), dtype=int)

    # Generate initial population randomly
    for i in range(population_size):
        population[i, :] = np.random.randint(2, size=n)
    
    # Calculate fitness for each individual in the population
    fitness = calculate_fitness(population)

    # Main loop
    for gen in range(num_generations):
        for i in range(population_size):
            # Select parents for reproduction using tournament selection
            parent1 = selection(population, fitness, 2)
            parent2 = selection(population, fitness, 2)

            # Perform crossover and mutation to create new offspring
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)

            # Mutation
            if np.random.rand() < mutation_rate:
                offspring1 = mutation(offspring1)
            if np.random.rand() < mutation_rate:
                offspring2 = mutation(offspring2)
            
            # Offsprings and Evaluation
            new_solutions = np.array([offspring1, offspring2])
            new_fitness = calculate_fitness(new_solutions)

            # Add offsprings into population
            population = np.vstack((population, new_solutions))
            fitness = np.append(fitness, new_fitness)

            # Delete worse solution from population 
            population, fitness = update_population(population, fitness, population_size)

            # Find the best fitness and corresponding solution
            best_fitness = np.max(fitness)
            best_index = np.argmax(fitness)
            best_solution = population[best_index, :]

        # Display generation information
        if gen % 10 == 0:
            print(f'Generation {gen}:')
            print(f'   Pop Size {population.shape[0]}:')
            print(f'   Best Fitness: {best_fitness:.2f}')
            print(f'   Best Solution: {best_solution}\n')

    print('### Genetic Algorithm Finished!')
    
    # Display the best solution and best fitness
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    return best_solution, best_fitness

# Function for population updating 
def update_population(population, fitness, population_size):
    # Ensure that the population does not exceed the desired population size
    while population.shape[0] > population_size:
        # Find the index of the worst solution 
        worst_index = np.argmin(fitness)
        
        # Remove the worst solution from the population and fitness array
        population = np.delete(population, worst_index, axis=0)
        fitness = np.delete(fitness, worst_index)

    return population, fitness

# Function for crossover (Single Point Crossover)
def crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = np.random.randint(1, n)
    offspring = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
    return offspring

# Function for mutation (swap two random cities)
def mutation(child):
    num_cities = len(child)
    mutated_child = child.copy()

    # Randomly select two positions to swap
    swap_pos1, swap_pos2 = np.random.choice(num_cities, 2, replace=False)

    # Swap the cities
    mutated_child[swap_pos1], mutated_child[swap_pos2] = mutated_child[swap_pos2], mutated_child[swap_pos1]

    return mutated_child

# Function for parents selection using tournament selection
def selection(population, fitness, tournament_size):
    # Get the number of individuals in the population
    num_individuals = population.shape[0]
    # Initialize the selected individual with zeros
    selected_individual = np.zeros(population.shape[1], dtype=int)

    # Initialize the best fitness value to zero
    best_fitness = 0
    # Loop over the number of tournaments
    for _ in range(tournament_size):
        # Select a random individual from the population
        random_index = np.random.randint(num_individuals)
        # Check if the randomly selected individual has better fitness
        if fitness[random_index] > best_fitness:  # select maximum
            # Update the best fitness value
            best_fitness = fitness[random_index]
            # Update the selected individual to the new best individual
            selected_individual = population[random_index, :]

    return selected_individual

# ====================================================================
# Example of solving container loading problem with genetic algorithm (GA).
# ====================================================================

# # Container loading instance case 1
# widths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # set widths of items (integer)
# lengths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # set lengths of items (integer)
# L = 10  # Set the length of container (integer)
# W = 5   # Set the width of container (integer)

# # Container loading instance case 2
# widths = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9])  # set widths of items (integer)
# lengths = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9])  # set lengths of items (integer)
# L = 20  # Set the length of container (integer)
# W = 10   # Set the width of container (integer)

# Container loading instance case 3
widths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # set widths of items (integer)
lengths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # set lengths of items (integer)
L = 10  # Set the length of container (integer)
W = 10   # Set the width of container (integer)
# ====================================================================
p_w = 1  # Weight penalty for excess weight
positions = []  # List to store positions of items

# Display the instance settings
print('### Container Loading Problem Instance Settings:')
print(f'    Item lengths: {lengths}')
print(f'    Item widths: {widths}')
print(f'    Container Length: {L}')
print(f'    Container Width: {W}')

# Parameter settings for GA
population_size = 50  # population size of GA
num_generations = 50  # number of generations of GA
mutation_rate = 0.05  # mutation rate of GA

# Function to calculate the fitness of each individual in the population
def calculate_fitness(population):
    global widths, lengths, L, W, p_w, positions
    population_size = population.shape[0] # Get the population size
    fitness = np.zeros(population_size) # Initialize the fitness array
    
    for i in range(population_size):
        selected_items = population[i, :].astype(bool) # Convert to boolean mask
        selected_widths = widths[selected_items] # Get the widths of selected items
        selected_lengths = lengths[selected_items] # Get the lengths of selected items

        # Sort the selected items according to the areas (i.e., width * length) in decreasing order
        areas = selected_widths * selected_lengths # Calculate the areas of selected items
        sorted_indices = np.argsort(areas)[::-1] # Sort the areas in decreasing order
        sorted_selected_widths = selected_widths[sorted_indices] # Sort the selected widths
        sorted_selected_lengths = selected_lengths[sorted_indices] # Sort the selected lengths

        # Load the items one by one into the container, which has a length L and a width W    
        container = np.zeros((L, W)) # Initialize the container
        positions = []  # List to store position_start and position_end
        
        for j in range(len(sorted_selected_widths)): # Loop through the selected items
            item_width = sorted_selected_widths[j] # Get the width of the item
            item_length = sorted_selected_lengths[j] # Get the length of the item

            # Find the position to place the item using first fit and corner point heuristics
            fit, position_start, position_end = find_position(container, item_width, item_length)

            # If the item fits, place it in the container
            if fit:
                container[position_start[0]:position_end[0], position_start[1]:position_end[1]] = 1 # Place the item
                positions.append([position_start, position_end])  # Store position_start and position_end

        # Get the fitness, the fitness is the loading rate (i.e., total area of loaded items / area of container)
        loaded_area = np.sum(container) # Calculate the total area of loaded items
        fitness[i] = loaded_area / (L * W) # Calculate the loading rate

        # Get the penalty, the penalty = p_w * total area of items that have not been loaded
        unloaded_area = np.sum(areas) - loaded_area # Calculate the total area of items that have not been loaded
        penalty = p_w * unloaded_area # Calculate the penalty
        fitness[i] -= penalty # Update the fitness by subtracting the penalty

    return fitness

# Function to find the position to place an item into a container using first fit and corner point heuristics
def find_position(container, item_width, item_length):
    container_width = container.shape[1] # Get the width of the container
    container_length = container.shape[0] # Get the length of the container
    fit = False  # flag indicating if the item fits in the container
    position_start = [0, 0]  # position to place the item
    position_end = [0, 0]  # end point of position

    # First fit heuristic
    for i in range(container_length - item_length + 1): # loop through container rows
        for j in range(container_width - item_width + 1): # loop through container columns
            if np.all(container[i:i+item_length, j:j+item_width] == 0): # check if the item fits in the container
                fit = True # set flag to True
                position_start = [i, j] # set position to place the item
                position_end = [i+item_length, j+item_width] # set end point of position

    return fit, position_start, position_end

# Function to plot the solution
def plot_solution(positions, L, W):
    plt.figure()
    # plt.axis('equal')
    plt.xlim(0, L)
    plt.ylim(0, W)
    # plt.tight_layout()
    # Plot container boundary
    container_boundary = np.array([[0,0], [L , 0], [L, W ], [0, W], [0, 0]])
    plt.plot(container_boundary[:, 0], container_boundary[:, 1], 'k', linewidth=2)

    # Plot item boundaries
    num_items = len(positions)
    colors = plt.cm.hsv(np.linspace(0.1, 1, num_items))

    for k in range(num_items):
        item_start = positions[k][0]
        item_end = positions[k][1]
        item_boundary = np.array([[item_start[0], item_start[1]], [item_end[0], item_start[1]],
                                 [item_end[0], item_end[1]], [item_start[0], item_end[1]], [item_start[0], item_start[1]]])
        plt.fill(item_boundary[:, 0], item_boundary[:, 1], color=colors[k])

    # Create a list to store the item names
    item_names = ['Container'] + [f'Cargo {i+1}' for i in range(num_items)]

    # Add the legend with the item names and corresponding colors
    plt.legend(item_names, loc='upper left', bbox_to_anchor=(1, 1))

    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.title('Container Loading')
    plt.tight_layout()
    plt.show()


# Run the genetic algorithm and obtain the best solution and fitness
best_solution, best_fitness = GA_01(population_size, num_generations, mutation_rate, len(widths), calculate_fitness)

# Plot the best solution
fitness = calculate_fitness(np.array([best_solution]))
plot_solution(positions, L, W)