import random
import numpy as np
import matplotlib.pyplot as plt

# Define the coordinates of the cities
cities = np.array([
    (1, 1),   # City 1
    (1, 2),   # City 2
    (3, 4),   # City 3
    (4, 2),   # City 4
    (5, 6),   # City 5
    (6, 3),   # City 6
    (7, 5)    # City 7
])
# Function to calculate the Euclidean distance between two cities
def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Function to calculate the total distance of the tour
def evaluate(individual):
    distance = 0
    for i in range(len(individual)):
        city1 = cities[individual[i % len(individual)]]
        city2 = cities[individual[(i + 1) % len(individual)]]
        distance += calculate_distance(city1, city2)
    return distance

# Initial population generator
def init_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

# Tournament selection
def select(population, fitnesses, k=3):
    tournament = random.sample(list(zip(population, fitnesses)), k)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]

# Order crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child_p1 = parent1[start:end]
    
    child = [item for item in parent2 if item not in child_p1]
    return child[:start] + child_p1 + child[start:]

# Mutate by swapping two cities
def mutate(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Main GA function
def genetic_algorithm(cities, pop_size=100, generations=500, crossover_prob=0.7, mutation_rate=0.01):
    num_cities = len(cities)
    population = init_population(pop_size, num_cities)
    
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [evaluate(ind) for ind in population]
        
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
    best_individual = min(population, key=evaluate)
    best_route = [cities[i] for i in best_individual]
    best_route.append(cities[best_individual[0]])  # Return to the start
    return best_individual, evaluate(best_individual), best_route

# Plot route
def plot_route(route):
    plt.figure(figsize=(8, 6))
    plt.plot([c[0] for c in route], [c[1] for c in route], '-o')
    plt.title('Traveling Salesman Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for index, city in enumerate(route[:-1]):
        plt.annotate(f'{index + 1}', (city[0], city[1]))
    plt.show()

if __name__ == "__main__":
    # Run the genetic algorithm to find the optimal route
    best_individual, best_distance, best_route = genetic_algorithm(cities)
    print("Best individual (route order): ", best_individual)
    print("Total distance of best route: ", best_distance)
    
    # Plot the best route found
    plot_route(best_route)