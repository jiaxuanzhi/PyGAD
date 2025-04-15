import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithmFSSP:
    def __init__(self, num_jobs, num_machines, population_size, num_generations, mutation_rate, processing_times):
        """
        Initialization of parameters for the Genetic Algorithm:
        param num_jobs: Number of jobs
        param num_machines: Number of machines
        param population_size: Population size
        param num_generations: Number of generations
        param mutation_rate: Probability of mutation
        param processing_times: Processing times matrix (jobs x machines)
        """
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.processing_times = processing_times

    def generate_initial_population(self):
        """
        Generate the initial population, where each individual is a permutation of job indices.
        """
        population = []
        for _ in range(self.population_size):
            individual = np.random.permutation(self.num_jobs)
            population.append(individual)
        return np.array(population)

    def calculate_makespan(self, order):
        """
        Calculate the makespan (total completion time) for a given job order.
        :param order: An array representing the order in which jobs are processed.
        :return: The makespan corresponding to this order.
        """
        times = np.zeros(self.num_machines) # Initialize the starting time for each machine
        for job in order: # Iterate through each job in the order
            times[0] += self.processing_times[job, 0] # Update the starting time for the first machine
            for machine in range(1, self.num_machines): # Iterate through the remaining machines
                if times[machine] < times[machine - 1]: # Check if the current machine is idle
                    times[machine] = times[machine - 1] # Update the starting time for the current machine
                times[machine] += self.processing_times[job, machine] # Update the completion time for the current job
        if order[0] != 0: # Check if the first job is not the first job in the order
            penalty = 100 # Set a penalty for the case where the first job is not the first job
            makespan = np.max(times) + penalty # Return the maximum time from the times array
        else:
            makespan = np.max(times) # Return the maximum time from the times array
        return makespan

    def evaluate_fitness(self, population):
        """
        Calculate the fitness (makespan) for each individual in the population.
        :param population: The current population.
        :return: An array of makespan values (the smaller the better).
        """
        fitness_values = np.zeros(len(population))
        for i, individual in enumerate(population):
            fitness_values[i] = self.calculate_makespan(individual)
        return fitness_values

    def selection_tournament(self, population, fitness, tournament_size=2):
        """
        Tournament selection: randomly choose several individuals and pick the one with the minimum makespan.
        :param population: The current population.
        :param fitness: Array of fitness (makespan values).
        :param tournament_size: Number of individuals to be selected in the tournament.
        :return: The chosen individual (job permutation).
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
        Ordered Crossover (OX) implementation.
        :param parent1: The first parent.
        :param parent2: The second parent.
        :return: The offspring resulting from OX crossover.
        """
        num_jobs = len(parent1)
        child = np.zeros(num_jobs, dtype=int)

        # Randomly determine the start and end positions for the crossover segment
        subset_start = np.random.randint(0, num_jobs - 1)
        subset_end = np.random.randint(subset_start + 1, num_jobs)

        # Copy the segment from parent1 to the child
        child[subset_start:subset_end] = parent1[subset_start:subset_end]

        # Fill out the remaining positions in the child using the parent2 sequence
        pointer = 0
        for job in parent2:
            if job not in child:
                while child[pointer] != 0:
                    pointer += 1
                child[pointer] = job
        return child

    def mutate(self, child):
        """
        Mutation operation: randomly swap two jobs with a certain probability.
        :param child: The offspring.
        :return: The mutated child.
        """
        if np.random.rand() < self.mutation_rate:
            swap_pos1 = np.random.randint(len(child))
            swap_pos2 = np.random.randint(len(child))
            child[swap_pos1], child[swap_pos2] = child[swap_pos2], child[swap_pos1]
        return child

    def update_population(self, population, fitness):
        """
        Keep the population size at 'population_size' by removing the worst individuals (those with the largest makespan).
        :param population: The expanded population (after offspring are added).
        :param fitness: Corresponding fitness values.
        :return: The updated population and fitness arrays of size 'population_size'.
        """
        while len(population) > self.population_size:
            worst_index = np.argmax(fitness)
            population = np.delete(population, worst_index, axis=0)
            fitness = np.delete(fitness, worst_index, axis=0)
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
        population = self.generate_initial_population()
        fitness = self.evaluate_fitness(population)

        best_makespan = np.inf
        best_individual = None

        for gen in range(self.num_generations):
            for _ in range(self.population_size):
                # Select parents using tournament selection
                parent1 = self.selection_tournament(population, fitness, 2)
                parent2 = self.selection_tournament(population, fitness, 2)

                # Create offspring using crossover and mutation
                offspring1 = self.crossover_OX(parent1, parent2)
                offspring2 = self.crossover_OX(parent2, parent1)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                # Insert new offspring into population
                population = np.vstack((population, offspring1))
                population = np.vstack((population, offspring2))

                # Compute fitness for the new offspring
                fit1 = self.calculate_makespan(offspring1)
                fit2 = self.calculate_makespan(offspring2)
                fitness = np.append(fitness, fit1)
                fitness = np.append(fitness, fit2)

                # Keep the population size consistent
                population, fitness = self.update_population(population, fitness)

                # Update the current best individual
                current_best = np.min(fitness)
                if current_best < best_makespan:
                    best_makespan = current_best
                    best_individual = population[np.argmin(fitness)]

            if (gen + 1) % 10 == 0:
                print(f"    Generation {gen + 1}:")
                print(f"   Pop Size {len(population)}:")
                print(f"       Best Fitness: {best_makespan:.2f}")
                print(f"       Best Solution: {best_individual}")

        print("### Genetic Algorithm Finished!")
        print("Best Sequence:")
        print(best_individual)
        print("Total Makespan:")
        print(best_makespan)
        return best_individual, best_makespan

# ====================================================================
# Example of solving FSSP using genetic algorithm (GA) with sequence encoding.
# ====================================================================

# Processing times matrix (jobs x machines)
processing_times = np.array([
    [3, 5],[5, 3],[4, 1]
]) 
num_jobs = processing_times.shape[0]    # Number of jobs
num_machines = processing_times.shape[1]    # Number of machines

# Parameter settings
population_size = 50   # Population size
num_generations = 50    # Number of generations
mutation_rate = 0.05    # Probability of mutation

# Run the genetic algorithm
ga_fssp = GeneticAlgorithmFSSP(num_jobs, num_machines, population_size, num_generations, mutation_rate, processing_times)  # Initialize GA
best_individual, best_makespan = ga_fssp.run()  # Run GA