import numpy as np
import matplotlib.pyplot as plt

def GA_01(population_size, num_generations, mutation_rate, length, calculate_fitness):
    """
    Main Genetic Algorithm function for container loading optimization.
    Implements selection, crossover, mutation, and population update operations.
    Args:
        population_size: Number of individuals in each generation
        num_generations: Total number of generations to evolve
        mutation_rate: Probability of mutation for each offspring
        length: Length of chromosome (solution representation)
        calculate_fitness: Function to evaluate solution fitness
    Returns:
        tuple: (best_solution, best_fitness) found during evolution
    """
    
    print('### Genetic Algorithm Settings:')
    print(f'    Population Size: {population_size}')
    print(f'    Number of Generations: {num_generations}')
    print(f'    Mutation Rate: {mutation_rate}')
    print('### Genetic Algorithm Start ...')

    # Initialize population
    population = np.array([np.random.permutation(length) for _ in range(population_size)])
    
    # Evaluate initial population fitness
    fitness = calculate_fitness(population)

    for gen in range(num_generations):
        for i in range(population_size):
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)

            if np.random.rand() < mutation_rate:
                parent1 = mutation(parent1)
                parent2 = mutation(parent2)

            # Create offspring through crossover
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)

            # Evaluate new offspring fitness
            new_solutions = np.array([offspring1, offspring2])
            new_fitness = calculate_fitness(new_solutions)

            # Add offspring to population
            population = np.vstack((population, new_solutions))
            fitness = np.append(fitness, new_fitness)

            # Maintain population size by removing worst individuals
            population, fitness = update_population(population, fitness, population_size)

            # Track best solution found so far
            best_fitness = np.max(fitness)
            best_index = np.argmax(fitness)
            best_solution = population[best_index, :]

        if gen % 10 == 0:
            print(f'Generation {gen}:')
            print(f'   Pop Size {population.shape[0]}:')
            print(f'   Best Fitness: {best_fitness:.2f}')
            print(f'   Best Solution: {best_solution}\n')

    print('### Genetic Algorithm Finished!')
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    return best_solution, best_fitness

def update_population(population, fitness, population_size):
    while population.shape[0] > population_size:
        worst_index = np.argmin(fitness)
        population = np.delete(population, worst_index, axis=0)
        fitness = np.delete(fitness, worst_index)
    return population, fitness

def crossover(parent1, parent2):
    n = len(parent1)
    crossover_point = np.random.randint(1, n)
    offspring = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
    return offspring

def mutation(child):
    num_items = len(child)
    mutated_child = child.copy()
    swap_pos1, swap_pos2 = np.random.choice(num_items, 2, replace=False)
    mutated_child[swap_pos1], mutated_child[swap_pos2] = mutated_child[swap_pos2], mutated_child[swap_pos1]
    return mutated_child

def selection(population, fitness):
    tournament_size = 3
    selected = np.random.choice(population.shape[0], tournament_size, replace=False)
    selected_fitness = fitness[selected]
    best_idx = np.argmax(selected_fitness)
    return population[selected[best_idx]]

def plot_solution(solution):
    fig, ax = plt.subplots()
    current_x = 0
    current_y = 0
    current_row_height = 0

    for item_idx in solution:
        item_width = widths[item_idx]
        item_length = lengths[item_idx]

        if current_x + item_width > container_width:
            current_x = 0
            current_y += current_row_height
            current_row_height = 0
        
        if current_y + item_length <= container_length:
            rect = plt.Rectangle((current_x, current_y), item_width, item_length, fill=True, edgecolor='r', linewidth=2)
            ax.add_patch(rect)
            current_x += item_width
            current_row_height = max(current_row_height, item_length)

    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_length)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('2D Container Loading')
    plt.show()


# Container and items description
widths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # widths of items
lengths = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # lengths of items
container_width = 10   # the width of the container
container_length = 10  # the length of the container
L = container_length  # Set the length of container (integer)
W = container_width   # Set the width of container (integer)
# ====================================================================
p_w = 1  # Weight penalty for excess weight
positions = []  # List to store positions of items

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

if __name__ == "__main__":
    # Execute Genetic Algorithm
    population_size = 100
    num_generations = 100
    mutation_rate = 0.1
    best_solution, _ = GA_01(population_size, num_generations, mutation_rate, len(widths), calculate_fitness)
    
    # Plot the results
    plot_solution(best_solution)