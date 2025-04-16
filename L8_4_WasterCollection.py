import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class GeneticAlgorithmWC:
    """
    Use a Genetic Algorithm to solve the Waste Collection (WC) problem.
    """
    def __init__(self, n_houses, population_size, num_generations, mutation_rate,
                house_locations, house_demands, capacity, depot_location):
        """
        Parameter Description:
        :param n_houses: Number of houses
        :param population_size: Size of the population
        :param num_generations: Number of generations
        :param mutation_rate: Probability of mutation
        :param house_locations: An array of (x, y) coordinates for each house, shape = [n_houses, 2]
        :param house_demands: An array of demands for each house, shape = [n_houses]
        :param capacity: Truck capacity
        :param depot_location: The (x, y) coordinate of the depot (parking station)
        """
        self.n_houses = n_houses
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

        self.house_locations = house_locations
        self.house_demands = house_demands
        self.capacity = capacity
        self.depot_location = depot_location

        # Build the distance matrix among houses
        self.distance_matrix = squareform(pdist(house_locations, 'euclidean'))
        # Build the distance from the depot to each house ([n_houses] array)
        dx = house_locations[:, 0] - depot_location[0]
        dy = house_locations[:, 1] - depot_location[1]
        self.distance_matrix_depot = np.sqrt(dx**2 + dy**2)
        # self.distance_matrix_depot = squareform(pdist(house_locations - depot_location, 'euclidean'))

    def generate_initial_population(self):
        """
        Generate the initial population, where each individual is a permutation of house indices.
        """
        population = []
        for _ in range(self.population_size):
            individual = np.random.permutation(self.n_houses)
            population.append(individual)
        return np.array(population)

    def calculate_wc_distance(self, house_sequence):
        """
        Calculate the total travel distance for the given house visiting sequence 
        in the waste collection problem. If the truck runs out of capacity for the next house, 
        it returns to the depot, and then continues from the depot again.
        """
        total_distance = 0.0
        rest_capacity = self.capacity - self.house_demands[house_sequence[0]]
        # Start: from depot to the first house
        total_distance += self.distance_matrix_depot[house_sequence[0]]

        for i in range(len(house_sequence) - 1):
            from_house = house_sequence[i]
            to_house = house_sequence[i + 1]
            rest_capacity -= self.house_demands[to_house]

            # If capacity is exceeded, return to depot and then start again
            if rest_capacity < 0:
                total_distance += self.distance_matrix_depot[from_house]  # return to depot
                total_distance += self.distance_matrix_depot[to_house]    # depot to the next house
                rest_capacity = self.capacity - self.house_demands[to_house]
            else:
                # If there is enough capacity, go directly from house to house
                total_distance += self.distance_matrix[from_house, to_house]

        # Return to depot after the last house
        last_house = house_sequence[-1]
        total_distance += self.distance_matrix_depot[last_house]
        return total_distance

    def evaluate_fitness(self, population):
        """
        Compute the total distance (the smaller, the better) for each individual in the population.
        """
        fitness_values = np.zeros(len(population))
        for i, individual in enumerate(population):
            fitness_values[i] = self.calculate_wc_distance(individual)
        return fitness_values

    def selection_tournament(self, population, fitness, tournament_size=2):
        """
        Tournament selection: randomly pick several individuals and choose 
        the one with the smallest distance (the best individual).
        """
        num_individuals = len(population)
        best_fitness = np.inf
        selected_individual = None
        for _ in range(tournament_size):
            random_index = np.random.randint(num_individuals)
            if fitness[random_index] < best_fitness:
                best_fitness = fitness[random_index]
                selected_individual = population[random_index]
        return selected_individual

    def crossover_OX(self, parent1, parent2):
        """
        Ordered Crossover (OX).
        :param parent1: The first parent
        :param parent2: The second parent
        :return: Offspring (similar to TSP)
        """
        num_cities = len(parent1)
        child = np.zeros(num_cities, dtype=int)

        # Randomly determine the segment for crossover
        subset_start = np.random.randint(0, num_cities - 1)
        subset_end = np.random.randint(subset_start + 1, num_cities)

        # Copy the gene segment from parent1 into the child
        child[subset_start:subset_end] = parent1[subset_start:subset_end]

        # Fill in remaining genes with parent2
        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] != 0:
                    pointer += 1
                child[pointer] = city
        return child

    def mutate(self, child):
        """
        Mutation operation: randomly swap two house indices.
        """
        if np.random.rand() < self.mutation_rate:
            swap_pos1 = np.random.randint(len(child))
            swap_pos2 = np.random.randint(len(child))
            child[swap_pos1], child[swap_pos2] = child[swap_pos2], child[swap_pos1]
        return child

    def update_population(self, population, fitness):
        """
        Maintain the population size by removing the worst individuals (largest distance).
        """
        while len(population) > self.population_size:
            worst_index = np.argmax(fitness)
            population = np.delete(population, worst_index, axis=0)
            fitness = np.delete(fitness, worst_index, axis=0)
        return population, fitness

    def run(self):
        """
        Main process: run the genetic algorithm.
        """
        print("### Parameter Settings:")
        print(f"    Population Size: {self.population_size}")
        print(f"    Number of Generations: {self.num_generations}")
        print(f"    Mutation Rate: {self.mutation_rate}")
        print("### Genetic Algorithm Start ...")

        # Generate the initial population and evaluate fitness
        population = self.generate_initial_population()
        fitness = self.evaluate_fitness(population)

        best_cost = np.inf
        best_individual = None

        for gen in range(self.num_generations):
            # Generate offspring
            for _ in range(self.population_size):
                # Select parents
                parent1 = self.selection_tournament(population, fitness, 2)
                parent2 = self.selection_tournament(population, fitness, 2)

                # Crossover and mutation
                offspring1 = self.crossover_OX(parent1, parent2)
                offspring2 = self.crossover_OX(parent2, parent1)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                # Insert offspring into the population
                population = np.vstack((population, offspring1))
                population = np.vstack((population, offspring2))

                # Compute the fitness of new offspring
                fit1 = self.calculate_wc_distance(offspring1)
                fit2 = self.calculate_wc_distance(offspring2)
                fitness = np.append(fitness, fit1)
                fitness = np.append(fitness, fit2)

                # Control population size
                population, fitness = self.update_population(population, fitness)

                # Update global best
                current_best = np.min(fitness)
                if current_best < best_cost:
                    best_cost = current_best
                    best_individual = population[np.argmin(fitness)]

            # Print info every few generations
            if (gen + 1) % 10 == 0:
                print(f"    Generation {gen + 1}:")
                print(f"   Population Size: {len(population)}")
                print(f"       Current Best Distance: {best_cost:.2f}")
                print(f"       Best Sequence: {best_individual}")

        print("### Genetic Algorithm Finished!")
        print("Optimal House Sequence:")
        print(best_individual)
        print("Total Distance:")
        print(best_cost)
        return best_individual, best_cost

    def plot_routes(self, best_sequence):
        """
        Plot the traveling routes. The best sequence is split into multiple sub-routes 
        due to capacity constraints. Each sub-route is drawn separately.
        """
        x = self.house_locations[:, 0]
        y = self.house_locations[:, 1]

        plt.figure()
        plt.title('Waste Collection Routes')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.scatter(x, y, c='blue', marker='o', s=40, label='House')

        # Plot the depot as well
        plt.scatter(self.depot_location[0], self.depot_location[1], c='red', marker='s', s=100, label='Depot')

        # Annotate each house with its index
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, str(i), fontsize=12, ha='right')

        # Split the sequence
        sub_routes = []
        sub_route = [best_sequence[0]]
        rest_capacity = self.capacity - self.house_demands[best_sequence[0]]

        for i in range(len(best_sequence) - 1):
            curr_house = best_sequence[i]
            next_house = best_sequence[i + 1]
            rest_capacity -= self.house_demands[next_house]

            # If capacity is insufficient, draw current sub-route and start a new one
            if rest_capacity < 0:
                sub_routes.append(sub_route)
                sub_route = [next_house]
                rest_capacity = self.capacity - self.house_demands[next_house]
            else:
                sub_route.append(next_house)

        # Add the last sub-route
        sub_routes.append(sub_route)

        # Draw each sub-route
        for i, route in enumerate(sub_routes):
            self._draw_segment(route, i + 1)

        plt.legend()
        plt.show()

        # Print each sub-route
        print("### Sub-routes:")
        for i, route in enumerate(sub_routes):
            print(f"Route {i + 1}: {list(route)}")

    def _draw_segment(self, sub_route, route_number):
        """
        Helper function: plots the route segment by connecting the depot 
        and the houses in sub_route, then returning to the depot.
        """
        x = self.house_locations[:, 0]
        y = self.house_locations[:, 1]

        depot_x = self.depot_location[0]
        depot_y = self.depot_location[1]

        route_x = [depot_x, x[sub_route[0]]]
        route_y = [depot_y, y[sub_route[0]]]

        for i in range(len(sub_route) - 1):
            route_x.append(x[sub_route[i + 1]])
            route_y.append(y[sub_route[i + 1]])

        route_x.append(depot_x)
        route_y.append(depot_y)

        color = np.random.rand(3,)
        plt.plot(route_x, route_y, color=color, linewidth=2, label=f'Route {route_number}')

# ---------------- Example usage below; can be run directly ----------------
# Waste Collection example: 7 houses, capacity = 1000
n_house = 7 # Number of houses
# House locations (x, y) coordinates
house_locations = np.array([
[100, 100],[100, 350],[300, 400],[400, 100],[500, 600],[600, 300],[700, 500]
])

house_demands = np.array([100, 200, 300, 100, 200, 300, 400]) # Demands for each house
depot_location = np.array([0, 0]) # Depot location (x, y)
capacity = 2000 # Truck capacity

# GA parameters
population_size = 50
num_generations = 50
mutation_rate = 0.05

ga_wc = GeneticAlgorithmWC(n_houses=n_house,
                          population_size=population_size,
                          num_generations=num_generations,
                          mutation_rate=mutation_rate,
                          house_locations=house_locations,
                          house_demands=house_demands,
                          capacity=capacity,
                          depot_location=depot_location)

best_seq, best_dist = ga_wc.run()
ga_wc.plot_routes(best_seq)