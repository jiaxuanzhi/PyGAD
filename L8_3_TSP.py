import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class GeneticAlgorithmTSP:
    def __init__(self, num_cities, population_size, num_generations, mutation_rate, city_locations):
        """
        Initialization of parameters for the Genetic Algorithm:
        param num_cities: Number of cities
        param population_size: Population size
        param num_generations: Number of generations
        param mutation_rate: Probability of mutation
        param city_locations: (x,y) coordinates of cities
        """
        self.num_cities = num_cities
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.city_locations = city_locations
        # Construct the distance matrix
        self.distance_matrix = squareform(pdist(city_locations, 'euclidean')) # Pairwise Euclidean distances

    def generate_initial_population(self):
        """
        Generate the initial population, where each individual is a permutation of city indices.
        """
        population = [] # List to store the population
        for _ in range(self.population_size):
            individual = np.random.permutation(self.num_cities) # Random permutation of city indices
            population.append(individual) # Add individual to population
        return np.array(population)

    def calculate_distance(self, route):
        """
        Calculate the total distance of a given route.
        :param route: An array representing the order in which cities are visited.
        :return: The total distance corresponding to this route.
        """
        distance = 0 # Initialize total distance
        # Calculate the distance between each pair of consecutive cities
        for i in range(len(route) - 1):
            from_city = route[i] # Current city
            to_city = route[i + 1] # Next city
            distance += self.distance_matrix[from_city, to_city] # Add distance to total
        # Add the distance from the last city back to the first city
        distance += self.distance_matrix[route[-1], route[0]] 
        return distance

    def evaluate_fitness(self, population):
        """
        Calculate the fitness (total distance) for each individual in the population.
        :param population: The current population.
        :return: An array of total distances (the smaller the better).
        """
        fitness_values = np.zeros(len(population)) # Initialize fitness array
        for i, individual in enumerate(population): # Iterate over the population
            fitness_values[i] = self.calculate_distance(individual) # Calculate total distance
        return fitness_values

    def selection_tournament(self, population, fitness, tournament_size=2):
        """
        Tournament selection: randomly choose several individuals and pick the one with the minimum distance.
        :param population: The current population.
        :param fitness: Array of fitness (distances).
        :param tournament_size: Number of individuals to be selected in the tournament.
        :return: The chosen individual (city permutation).
        """
        num_individuals = len(population) # Number of individuals in the population
        best_fitness = np.inf   # Initialize best fitness to infinity
        selected_individual = None  # Initialize selected individual to None
        for _ in range(tournament_size):    # Repeat tournament selection process
            random_index = np.random.randint(num_individuals)   # Randomly select an individual
            if fitness[random_index] < best_fitness:    # Update best fitness if better fitness found
                best_fitness = fitness[random_index]    # Update best fitness
                selected_individual = population[random_index]  # Update selected individual
        return selected_individual

    def crossover_OX(self, parent1, parent2):
        """
        Ordered Crossover (OX) implementation.
        :param parent1: The first parent.
        :param parent2: The second parent.
        :return: The offspring resulting from OX crossover.
        """
        num_cities = len(parent1)   # Number of cities
        child = np.zeros(num_cities, dtype=int)     # Initialize child as array of zeros

        # Randomly determine the start and end positions for the crossover segment
        subset_start = np.random.randint(0, num_cities - 1) # Start position
        subset_end = np.random.randint(subset_start + 1, num_cities)    # End position

        # Copy the segment from parent1 to the child
        child[subset_start:subset_end] = parent1[subset_start:subset_end]

        # Fill out the remaining positions in the child using the parent2 sequence
        pointer = 0    # Initialize pointer
        for city in parent2:    # Iterate over each city in parent2
            if city not in child:   # If city not already in child
                while child[pointer] != 0:  # Move pointer to next empty position
                    pointer += 1    # Increment pointer
                child[pointer] = city   # Add city to child
        return child

    def mutate(self, child):
        """
        Mutation operation: randomly swap two cities with a certain probability.
        :param child: The offspring.
        :return: The mutated child.
        """
        if np.random.rand() < self.mutation_rate:   # Check if mutation should be performed
            swap_pos1 = np.random.randint(len(child))   # Randomly select two positions
            swap_pos2 = np.random.randint(len(child))   # Randomly select two positions
            child[swap_pos1], child[swap_pos2] = child[swap_pos2], child[swap_pos1] # Swap cities
        return child

    def update_population(self, population, fitness):
        """
        Keep the population size at 'population_size' by removing the worst individuals (those with the largest distance).
        :param population: The expanded population (after offspring are added).
        :param fitness: Corresponding fitness values.
        :return: The updated population and fitness arrays of size 'population_size'.
        """
        while len(population) > self.population_size:   # While population size exceeds desired size
            worst_index = np.argmax(fitness)    # Find the index of the worst individual
            population = np.delete(population, worst_index, axis=0)  # Remove worst individual from population
            fitness = np.delete(fitness, worst_index, axis=0)   # Remove corresponding fitness value
        return population, fitness

    def run(self):
        """
        The main procedure that runs the genetic algorithm.
        """
        print("### Parameter Settings:")
        print(f"    Population Size: {self.population_size}")
        print(f"    Number of Generations: {self.num_generations}")
        print(f"    Mutation Rate: {self.mutation_rate}")
        print("### Genetic Algorithm Start ...")

        # Generate the initial population and evaluate fitness
        population = self.generate_initial_population() # Generate initial population
        fitness = self.evaluate_fitness(population) # Evaluate fitness of initial population

        best_cost = np.inf  # Initialize best cost to infinity
        best_individual = None  # Initialize best individual to None

        for gen in range(self.num_generations):
            for _ in range(self.population_size):
                # Select parents using tournament selection
                parent1 = self.selection_tournament(population, fitness, 2) # Select first parent
                parent2 = self.selection_tournament(population, fitness, 2) # Select second parent

                # Create offspring using crossover and mutation
                offspring1 = self.crossover_OX(parent1, parent2)    # Perform crossover
                offspring2 = self.crossover_OX(parent2, parent1)    # Perform crossover
                offspring1 = self.mutate(offspring1)    # Perform mutation
                offspring2 = self.mutate(offspring2)    # Perform mutation

                # Insert new offspring into population
                population = np.vstack((population, offspring1))    # Add offspring1 to population
                population = np.vstack((population, offspring2))    # Add offspring2 to population

                # Compute fitness for the new offspring
                fit1 = self.calculate_distance(offspring1)  # Calculate distance for offspring1
                fit2 = self.calculate_distance(offspring2)  # Calculate distance for offspring2
                fitness = np.append(fitness, fit1)  # Append distance for offspring1
                fitness = np.append(fitness, fit2)  # Append distance for offspring2

                # Keep the population size consistent
                population, fitness = self.update_population(population, fitness)

                # Update the current best individual
                current_best = np.min(fitness)
                if current_best < best_cost:    # If new best solution found
                    best_cost = current_best    # Update best cost
                    best_individual = population[np.argmin(fitness)]    # Update best individual

            if (gen + 1) % 10 == 0:
                print(f"    Generation {gen + 1}:")
                print(f"   Pop Size {len(population)}:")
                print(f"       Best Fitness: {best_cost:.2f}")
                print(f"       Best Solution: {best_individual}")

        print("### Genetic Algorithm Finished!")
        print("Best Sequence:")
        print(best_individual)
        print("Total Cost:")
        print(best_cost)
        return best_individual, best_cost

    def plot_route(self, route):
        """
        Plot the route of the TSP solution.
        :param route: The best route (a permutation of city indices).
        """
        x_coords = self.city_locations[:, 0]    # x-coordinates of cities
        y_coords = self.city_locations[:, 1]    # y-coordinates of cities
        plt.figure()
        # Plot the cities
        plt.plot(x_coords, y_coords, 'bo', markersize=10)
        plt.title('Seven-city TSP')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        # Connect the cities in the order of the route
        for i in range(len(route) - 1): # Iterate over each city in the route
            from_city = route[i]   # Current city
            to_city = route[i + 1]  # Next city
            plt.plot([x_coords[from_city], x_coords[to_city]],
                    [y_coords[from_city], y_coords[to_city]],
                    'r-', linewidth=2)  # Connect the cities
        # Close the loop
        from_city = route[-1]   # Last city
        to_city = route[0]  # First city
        plt.plot([x_coords[from_city], x_coords[to_city]],
                [y_coords[from_city], y_coords[to_city]],
                'r-', linewidth=2)  # Connect the cities
        plt.show()

# ====================================================================
# Example of solving TSP using genetic algorithm (GA) with sequence encoding.
# ====================================================================
# City locations
city_locations = np.array([
    [1, 1], [1, 2], [3, 4], [4, 2], [5, 6], [6, 3], [7, 5]
])  # (x, y) coordinates of cities
num_cities = len(city_locations)    # Number of cities
# Parameter settings
population_size = 50    # Population size
num_generations = 50    # Number of generations
mutation_rate = 0.05    # Mutation rate

# Run the genetic algorithm
ga_tsp = GeneticAlgorithmTSP(num_cities, population_size, num_generations, mutation_rate, city_locations)  # Initialize GA
best_individual, best_distance = ga_tsp.run()   # Run the GA  

# Plot the result
ga_tsp.plot_route(best_individual)

