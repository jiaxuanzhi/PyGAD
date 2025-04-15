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
# Waste Collection example: 7 houses, capacity = 1000
n_house = 7
house_locations = np.array([
[100, 100],[100, 350],[300, 400],[400, 100],[500, 600],[600, 300],[700, 500]
])

house_demands = np.array([100, 200, 300, 100, 200, 300, 400])
depot_location = np.array([0, 0])
capacity = 2000

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