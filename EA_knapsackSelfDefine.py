# Import numpy library for array operations
import numpy as np

def GA_01(population_size, num_generations, mutation_rate, length, calculate_fitness):
    """
    Main Genetic Algorithm function for optimization problems.
    Implements selection, crossover, mutation and population update operations.
    
    Args:
        population_size: Number of individuals in each generation
        num_generations: Total number of generations to evolve
        mutation_rate: Probability of mutation for each offspring
        length: Length of chromosome (solution representation)
        calculate_fitness: Function to evaluate solution fitness
    
    Returns:
        tuple: (best_solution, best_fitness) found during evolution
    """
    
    # Print algorithm configuration parameters
    print('### Genetic Algorithm Settings:')
    print(f'    Population Size: {population_size}')
    print(f'    Number of Generations: {num_generations}')
    print(f'    Mutation Rate: {mutation_rate}')
    print('### Genetic Algorithm Start ...')

    # Initialize population array with zeros
    n = length  # Store chromosome length
    population = np.zeros((population_size, n), dtype=int)  # Create empty population array

    # Generate initial random population
    for i in range(population_size):
        # Create random binary chromosome for each individual
        population[i, :] = np.random.randint(2, size=n)
    
    # Evaluate initial population fitness
    fitness = calculate_fitness(population)  # Calculate fitness for all individuals

    # Main generational loop
    for gen in range(num_generations):
        # Process each individual in current population
        for i in range(population_size):
            # Parent selection using tournament selection
            parent1 = selection(population, fitness, 2)  # Select first parent
            parent2 = selection(population, fitness, 2)  # Select second parent

            # Create offspring through crossover
            offspring1 = crossover(parent1, parent2)  # First offspring
            offspring2 = crossover(parent2, parent1)  # Second offspring

            # Apply mutation with given probability
            if np.random.rand() < mutation_rate:  # Check mutation chance for offspring1
                offspring1 = mutation(offspring1)  # Mutate first offspring
            if np.random.rand() < mutation_rate:  # Check mutation chance for offspring2
                offspring2 = mutation(offspring2)  # Mutate second offspring
            
            # Evaluate new offspring fitness
            new_solutions = np.array([offspring1, offspring2])  # Combine offspring
            new_fitness = calculate_fitness(new_solutions)  # Calculate their fitness

            # Add offspring to population
            population = np.vstack((population, new_solutions))  # Vertically stack
            fitness = np.append(fitness, new_fitness)  # Append new fitness values

            # Maintain population size by removing worst individuals
            population, fitness = update_population(population, fitness, population_size)

            # Track best solution found so far
            best_fitness = np.max(fitness)  # Get current best fitness
            best_index = np.argmax(fitness)  # Get index of best individual
            best_solution = population[best_index, :]  # Extract best solution

        # Print progress every 10 generations
        if gen % 10 == 0:
            print(f'Generation {gen}:')
            print(f'   Pop Size {population.shape[0]}:')
            print(f'   Best Fitness: {best_fitness:.2f}')
            print(f'   Best Solution: {best_solution}\n')

    # Algorithm completion message
    print('### Genetic Algorithm Finished!')
    
    # Display final results
    print(f"Best solution: {best_solution}")  # Print best chromosome
    print(f"Best fitness: {best_fitness}")   # Print best fitness value

    return best_solution, best_fitness  # Return results

def update_population(population, fitness, population_size):
    """
    Maintains constant population size by removing worst individuals.   
    Args:
        population: Current population array
        fitness: Array of fitness values
        population_size: Target population size  
    Returns:
        tuple: (updated_population, updated_fitness)
    """
    # Remove individuals until target size is reached
    while population.shape[0] > population_size:
        worst_index = np.argmin(fitness)  # Find worst individual index
        
        # Remove worst individual from population and fitness array
        population = np.delete(population, worst_index, axis=0)
        fitness = np.delete(fitness, worst_index)

    return population, fitness

def crossover(parent1, parent2):
    """
    Single-point crossover between two parent chromosomes.   
    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome   
    Returns:
        New offspring chromosome
    """
    n = len(parent1)  # Get chromosome length
    # Select random crossover point (between 1 and n-1)
    crossover_point = np.random.randint(1, n)
    # Combine parent segments to create offspring
    offspring = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
    return offspring

def mutation(child):
    """
    Swap mutation operator - exchanges two random genes.   
    Args:
        child: Chromosome to mutate  
    Returns:
        Mutated chromosome
    """
    num_cities = len(child)  # Get chromosome length
    mutated_child = child.copy()  # Create copy to modify

    # Select two distinct random positions
    swap_pos1, swap_pos2 = np.random.choice(num_cities, 2, replace=False)

    # Perform swap mutation
    mutated_child[swap_pos1], mutated_child[swap_pos2] = mutated_child[swap_pos2], mutated_child[swap_pos1]

    return mutated_child

def selection(population, fitness, tournament_size):
    """
    Tournament selection for parent selection.
    
    Args:
        population: Current population
        fitness: Fitness values
        tournament_size: Number of candidates in tournament
    
    Returns:
        Selected parent chromosome
    """
    num_individuals = population.shape[0]  # Get population size
    selected_individual = np.zeros(population.shape[1], dtype=int)  # Initialize selected individual
    best_fitness = 0  # Initialize best fitness tracker

    # Run tournament selection
    for _ in range(tournament_size):
        random_index = np.random.randint(num_individuals)  # Random candidate
        # Update selection if better fitness found
        if fitness[random_index] > best_fitness:
            best_fitness = fitness[random_index] # Update best fitness
            selected_individual = population[random_index, :] # Update selected individual

    return selected_individual

# ====================================================================
# Example of solving Knapsack Problem with genetic algorithm (GA).
# ====================================================================
values = np.array([1000, 1500, 2000, 2500, 4000])  # Item values
weights = np.array([1, 2, 2, 3, 4])  # Item weights
max_weight = 7  # Maximum knapsack capacity
weight_penalty = 50000  # Penalty for exceeding capacity

# Print problem instance details
print('### Knapsack Problem Instance Settings:')
print(f'    Item Values: {values}')
print(f'    Item Weights: {weights}')
print(f'    Maximum Weight: {max_weight}')

# GA Parameters
population_size = 50  # Individuals per generation
num_generations = 50  # Total generations
mutation_rate = 0.05  # Mutation probability

def calculate_fitness(population):
    """
    Calculates fitness for knapsack solutions.
    Applies penalty for solutions exceeding weight limit. 
    Args:
        population: Current population
    Returns:
        Array of fitness values
    """
    global values, weights, max_weight, weight_penalty
    population_size = population.shape[0] # Get population size
    fitness = np.zeros(population_size) # Initialize fitness array 
    # Evaluate each individual
    for i in range(population_size):
        selected_items = population[i, :].astype(bool)  # Convert to boolean mask
        total_weight = np.sum(weights[selected_items])  # Calculate total weight
        fitness[i] = np.sum(values[selected_items])  # Initial fitness = total value       
        # Apply penalty if overweight
        if total_weight > max_weight:
            fitness[i] -= weight_penalty * (total_weight - max_weight) # Reduce fitness   
    return fitness

# Execute Genetic Algorithm
best_solution, best_fitness = GA_01(population_size, num_generations, mutation_rate, len(values), calculate_fitness)