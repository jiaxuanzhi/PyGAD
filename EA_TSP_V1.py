# A simple example of solving TSP using genetic algorithm (GA) with sequence encoding. Version 1.
# The city locations are given in 2D space. The fitness of an individual is the total distance of the route.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class GeneticAlgorithmTSP:
    def __init__(self, num_cities, population_size, num_generations, mutation_rate, city_locations):
        self.num_cities = num_cities
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.city_locations = city_locations
        self.distance_matrix = self.calculate_distance_matrix()
        
    def calculate_distance_matrix(self):
        return squareform(pdist(self.city_locations))

    def generate_initial_population(self):
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]

    def calculate_fitness(self, population):
        fitness = []
        for individual in population:
            total_distance = 0
            for i in range(self.num_cities - 1):
                total_distance += self.distance_matrix[individual[i], individual[i + 1]]
            total_distance += self.distance_matrix[individual[-1], individual[0]]  # Return to the start
            fitness.append(total_distance)
        return np.array(fitness)

    def select_parents_tournament(self, population, fitness, tournament_size=2):
        selected = []
        for _ in range(2):
            participants = np.random.choice(self.population_size, tournament_size)
            best_index = participants[np.argmin(fitness[participants])]
            selected.append(population[best_index])
        return selected

    def crossover_OX(self, parent1, parent2):
        start, end = sorted(np.random.choice(range(self.num_cities), 2, replace=False))
        child = [-1] * self.num_cities
        child[start:end] = parent1[start:end]

        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = city
        return np.array(child)

    def mutate(self, child):
        if np.random.rand() < self.mutation_rate:
            p1, p2 = np.random.choice(range(self.num_cities), 2, replace=False)
            child[p1], child[p2] = child[p2], child[p1]
        return child

    def update_population(self, population, fitness):
        indices = np.argsort(fitness)
        return [population[i] for i in indices[:self.population_size]], fitness[indices[:self.population_size]]

    def run(self):
        population = self.generate_initial_population()
        fitness = self.calculate_fitness(population)

        for gen in range(self.num_generations):
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents_tournament(population, fitness)
                offspring1 = self.mutate(self.crossover_OX(parent1, parent2))
                offspring2 = self.mutate(self.crossover_OX(parent2, parent1))
                new_population.extend([offspring1, offspring2])

            population.extend(new_population)
            fitness = self.calculate_fitness(population)
            population, fitness = self.update_population(population, fitness)

            if gen % 10 == 0:
                print(f"Generation {gen + 1}:")
                print(f"   Best Distance: {fitness[0]:.2f}")
                print(f"   Best Route: {population[0]}")

        print(f"Best Total Distance: {fitness[0]}")
        print(f"Best Solution: {population[0]}")
        return population[0], fitness[0]

    def plot_route(self, route):
        plt.figure()
        x, y = self.city_locations[:, 0], self.city_locations[:, 1]
        plt.plot(x, y, 'bo', markersize=10)
        for i in range(len(route)):
            start_city = route[i]
            end_city = route[(i + 1) % len(route)]
            plt.plot([x[start_city], x[end_city]], [y[start_city], y[end_city]], 'r-')
        plt.title('Traveling Salesman Problem')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

# ====================================================================
# Example of solving TSP using genetic algorithm (GA) with sequence encoding.
# ====================================================================

# City locations
city_locations = np.array([
    [1, 1], [1, 2], [3, 4], [4, 2], [5, 6], [6, 3], [7, 5]
])

# Parameters for GA
num_cities = city_locations.shape[0]
population_size = 50
num_generations = 50
mutation_rate = 0.05

# Initialize and run the genetic algorithm
ga_tsp = GeneticAlgorithmTSP(num_cities, population_size, num_generations, mutation_rate, city_locations)
best_route, best_distance = ga_tsp.run()
ga_tsp.plot_route(best_route)