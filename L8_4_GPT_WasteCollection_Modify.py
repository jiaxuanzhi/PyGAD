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

def plot_routes(best_sequence, house_locations, depot_location, house_demands, capacity):
    """
    Based on the best (complete) visiting sequence, split the entire route into multiple sub-routes.
    When the remaining capacity is insufficient to collect the garbage from the next house,
    the current sub-route is ended by returning to the depot; then a new sub-route starts.
    Each sub-route is plotted in a different color.
    """
    # Split the best sequence into multiple sub-routes
    sub_routes = []       # List to store multiple sub-routes; each sub-route contains a list of house indices
    sub_route = [best_sequence[0]]
    remaining_capacity = capacity - house_demands[best_sequence[0]]

    # Traverse the next houses in the sequence
    for i in range(len(best_sequence) - 1):
        current_house = best_sequence[i]
        next_house = best_sequence[i+1]
        
        if house_demands[next_house] > remaining_capacity:
            # Insufficient capacity for the next house: end the current sub-route by returning to the depot and save it
            sub_routes.append(sub_route)
            # Begin a new sub-route starting with the next house
            sub_route = [next_house]
            remaining_capacity = capacity - house_demands[next_house]
        else:
            # Continue the current sub-route
            sub_route.append(next_house)
            remaining_capacity -= house_demands[next_house]

    # Add the last sub-route
    sub_routes.append(sub_route)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.title('Waste Collection Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # Plot all houses
    x_all = house_locations[:, 0]
    y_all = house_locations[:, 1]
    plt.scatter(x_all, y_all, c='blue', marker='o', s=40, label='House')

    # Plot the depot, here represented by a red square
    plt.scatter(depot_location[0], depot_location[1], c='red', marker='s', s=100, label='Depot')

    # Annotate each house with its index (optional)
    for i, (xi, yi) in enumerate(zip(x_all, y_all)):
        plt.text(xi, yi, str(i), fontsize=10, ha='right')

    # For each sub-route, generate a random color and plot the route
    for idx, route in enumerate(sub_routes, start=1):
        # Construct the full route, starting from the depot, visiting the houses, and returning to the depot
        route_x = [depot_location[0]] + [house_locations[i][0] for i in route] + [depot_location[0]]
        route_y = [depot_location[1]] + [house_locations[i][1] for i in route] + [depot_location[1]]
        color = np.random.rand(3,)  # Generate a random color
        plt.plot(route_x, route_y, color=color, marker='o', linewidth=2, label=f'Route {idx}')

    plt.legend()
    plt.show()

    # Output the house indices for each sub-route
    print("### Sub-routes:")
    for idx, route in enumerate(sub_routes, start=1):
        print(f"Route {idx}: {route}")

if __name__ == "__main__":
    # Run the genetic algorithm to find the optimal route
    best_individual, best_distance, best_route = genetic_algorithm(house_locations)
    print("Best route order: ", best_individual)
    print("Total distance of best route: ", best_distance)
    
    # Plot the best route found
    # plot_route(best_route)
    plot_routes(best_individual, house_locations, depot_location, house_demands, capacity)