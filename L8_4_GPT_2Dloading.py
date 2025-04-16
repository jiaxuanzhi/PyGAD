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

def calculate_fitness(population):
    """
    Calculates fitness for placement solutions.
    Evaluates how well the items fit in the container.
    Args:
        population: Current population
    Returns:
        Array of fitness values
    """
    population_size = population.shape[0]
    fitness = np.zeros(population_size)
    
    for i in range(population_size):
        layout = population[:, 1]
        packed_width = np.zeros_like(layout, dtype=int)
        packed_length = np.zeros_like(layout, dtype=int)
        used_area = 0
        
        items = [(widths[idx], lengths[idx]) for idx in layout]
        
        W_current = 0  # Width in current row
        L_current = 0  # Length in current row
        
        for width, length in items:
            if W_current + width <= container_width:
                packed_width[i] = W_current + width
                packed_length[i] = L_current + length
                W_current += width
                used_area += width * length
            else:
                W_current = 0
                L_current += max(lengths[:i+1])
                
        fitness[i] = used_area / (container_width * container_length)
    
    return fitness

if __name__ == "__main__":
    # Execute Genetic Algorithm
    population_size = 50
    num_generations = 100
    mutation_rate = 0.05
    best_solution, _ = GA_01(population_size, num_generations, mutation_rate, len(widths), calculate_fitness)
    
    # Plot the results
    plot_solution(best_solution)