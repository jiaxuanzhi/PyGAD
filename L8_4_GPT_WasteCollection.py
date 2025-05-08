import random
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the waste collection problem
n_house = 7  # Number of houses
house_locations = np.array([
    [100, 100],
    [100, 350],
    [300, 400],
    [400, 100],
    [500, 600],
    [600, 300],
    [700, 500]
])  # House locations (x, y) coordinates
house_demands = np.array([100, 200, 300, 100, 200, 300, 400])  # Demands for each house
depot_location = np.array([0, 0])  # Depot location (x, y)
capacity = 2000  # Truck capacity

# Function to calculate the Euclidean distance between two locations
def calculate_distance(loc1, loc2):
    return np.linalg.norm(loc1 - loc2)

# Function to evaluate the total distance of the collection route
def evaluate(individual):
    total_distance = 0
    current_load = 0
    current_location = depot_location
    
    for i in range(len(individual)):
        house_index = individual[i]
        house_location = house_locations[house_index]
        
        # Calculate distance from current location to the next house
        total_distance += calculate_distance(current_location, house_location)
        
        # Update current load and check capacity
        current_load += house_demands[house_index]
        if current_load > capacity:
            # Return to depot and update current load
            total_distance += calculate_distance(house_location, depot_location)
            current_load = house_demands[house_index]
        
        # Move to the next house
        current_location = house_location
    
    # Return to depot from the last house
    total_distance += calculate_distance(current_location, depot_location)
    
    return total_distance

# Initial population generator
def init_population(pop_size, num_houses):
    population = []
    for _ in range(pop_size):
        individual = list(range(num_houses))
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

# Mutate by swapping two houses
def mutate(individual, mutation_rate=0.01):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Main GA function
def genetic_algorithm(house_locations, pop_size=100, generations=500, crossover_prob=0.7, mutation_rate=0.01):
    num_houses = len(house_locations)
    population = init_population(pop_size, num_houses)
    
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
    best_route = [depot_location] + [house_locations[i] for i in best_individual] + [depot_location]
    return best_individual, evaluate(best_individual), best_route

# Plot route
import matplotlib.pyplot as plt
import numpy as np

def plot_route(route):
    plt.figure(figsize=(8, 6))
    plt.plot([loc[0] for loc in route], [loc[1] for loc in route], '-o')
    plt.title('Waste Collection Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    for index, loc in enumerate(route[1:-1], start=1):
        plt.annotate(f'{index}', (loc[0], loc[1]))
    plt.scatter(depot_location[0], depot_location[1], color='red', label='Depot')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Run the genetic algorithm to find the optimal route
    best_individual, best_distance, best_route = genetic_algorithm(house_locations)
    print("Best route order: ", best_individual)
    print("Total distance of best route: ", best_distance)
    
    # Plot the best route found
    # plot_route(best_route)
    plot_route(best_route)