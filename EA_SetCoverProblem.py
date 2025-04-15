import numpy as np

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
# Example of solving Set Cover with genetic algorithm (GA).
# ====================================================================

# Set cover instance
universe = {1, 2, 3, 4, 5}  # Universe of elements
sets = [
    {1, 2, 3},
    {2, 4},
    {3, 4},
    {4, 5}
]  # Collection of sets

# Display the instance settings
print('### Set Cover Problem Instance Settings:')
print(f'    Item universe: {universe}')
sets_str = ', '.join([str(s) for s in sets])
print(f'    Item sets: {sets_str}')

# Parameter settings for GA
population_size = 50  # population size of GA
num_generations = 50  # number of generations of GA
mutation_rate = 0.05  # mutation rate of GA

# Function to calculate the fitness of each individual in the population
def calculate_fitness(population):
    global sets, universe
    population_size = population.shape[0]
    fitness = np.zeros(population_size)
    for i in range(population_size):
        covered = set()  # Initialize covered universe as empty
        for j in range(len(sets)):
            if population[i, j] == 1:
                covered.update(sets[j])  # Union of covered elements and elements of the current set
        uncovered_items = universe - covered  # Find uncovered items
        penalty = 100 * len(uncovered_items)  # Number of uncovered items
        num_sets_used = np.sum(population[i, :])  # Calculate the number of sets used
        # Calculate fitness as the number of sets used plus the number of uncovered items
        fitness[i] = -(num_sets_used + penalty)
    return fitness

# Run the genetic algorithm and obtain the best solution and fitness
best_solution, best_fitness = GA_01(population_size, num_generations, mutation_rate, len(sets), calculate_fitness)