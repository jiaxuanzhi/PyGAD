import random
import numpy as np
import matplotlib.pyplot as plt

# Processing times for each job on each machine
# T[i][j] -> Time taken by job i on machine j
processing_times = np.array([
    [3, 4],  # Job 1
    [3, 3],  # Job 2
    [4, 1]   # Job 3
])

num_jobs = processing_times.shape[0]
num_machines = processing_times.shape[1]

# Function to calculate the makespan of a job sequence
def calculate_makespan(sequence):
    num_jobs = len(sequence)
    completion_time = np.zeros((num_jobs, num_machines))
    
    # Calculate the completion times for the first machine
    completion_time[0, 0] = processing_times[sequence[0], 0]
    for i in range(1, num_jobs):
        completion_time[i, 0] = completion_time[i-1, 0] + processing_times[sequence[i], 0]
    
    # Calculate the completion times for the remaining machines
    for j in range(1, num_machines):
        # First job on machine j
        completion_time[0, j] = completion_time[0, j-1] + processing_times[sequence[0], j]
        for i in range(1, num_jobs):
            completion_time[i, j] = max(completion_time[i-1, j], completion_time[i, j-1]) + processing_times[sequence[i], j]
    
    return completion_time[-1, -1]  # Return the makespan

# Initialize the population
def init_population(pop_size):
    population = []
    base_individual = list(range(num_jobs))
    for _ in range(pop_size):
        individual = base_individual[:]
        random.shuffle(individual)
        population.append(individual)
    return population

# Tournament selection
def select(population, fitnesses, k=3):
    tournament = random.sample(list(zip(population, fitnesses)), k)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]

# Partially-Mapped Crossover (PMX)
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [-1] * size
    # Copy the slice from parent1 to child
    for i in range(start, end + 1):
        child[i] = parent1[i]

    # Fill in the remainder from parent2 while preserving the order 
    for i in range(size):
        if child[i] == -1:
            candidate = parent2[i]
            while candidate in child:
                candidate = parent2[parent1.index(candidate)]
            child[i] = candidate

    return child

# Mutation by swapping two jobs
def mutate(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Main GA function
def genetic_algorithm(pop_size=100, generations=500, crossover_prob=0.7, mutation_rate=0.01):
    population = init_population(pop_size)
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_makespan(ind) for ind in population]
        
        # Create new population
        new_population = []

        for _ in range(pop_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            
            # Crossover
            if random.random() < crossover_prob:
                offspring1 = crossover(parent1, parent2)
                offspring2 = crossover(parent2, parent1)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]
            
            # Mutate
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            
            new_population.extend([offspring1, offspring2])
        
        population = new_population
    
    # Get best solution
    best_individual = min(population, key=calculate_makespan)
    best_makespan = calculate_makespan(best_individual)
    
    return best_individual, best_makespan

if __name__ == "__main__":
    # Run the genetic algorithm to find the optimal job sequence
    best_sequence, best_makespan = genetic_algorithm()
    
    print("Optimal job sequence: ", best_sequence)
    print("Minimum makespan: ", best_makespan)