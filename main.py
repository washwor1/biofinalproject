import numpy as np
from deap import base, creator, tools, algorithms
import time
import csv
import pandas as pd
import model as m

# Initialize DEAP toolbox
toolbox = base.Toolbox()
input_shape = (880,1000,4)
# Define the fitness function and individual type
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0))  # Maximizing fitness function
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Define the individual and population initialization functions
def create_individual():
    return m.create_cnn_lstm_model(input_shape)

def create_population(n):
    return [create_individual() for _ in range(n)]

toolbox.register("individual", create_individual)
toolbox.register("population", create_population)

# Define the evaluation function
def evaluate_individual(individual):
    fitness, fail = pg.play_game(individual)
    return fitness, fail

toolbox.register("evaluate", evaluate_individual)

# Define the selection, mutation, and crossover operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

# Initialize the population
population = toolbox.population(n=10)

# Prepare statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("median", np.median, axis=0)
stats.register("max", np.max, axis=0)

# Prepare CSV file
with open('stats.csv', 'w', newline='') as csvfile:
    fieldnames = ['generation', 'avg_fitness', 'median_fitness', 'max_fitness', 'failure_rate', 'time_elapsed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Evolutionary loop
for gen in range(10):
    start_time = time.time()
    
    # Perform the genetic operations
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = toolbox.map(toolbox.evaluate, offspring)
    
    # Update the fitness values
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    
    # Gather statistics
    record = stats.compile(population)
    
    # Print the stats
    print(f"Generation {gen}:")
    print(f"  Avg Fitness: {record['avg'][0]}")
    print(f"  Median Fitness: {record['median'][0]}")
    print(f"  Max Fitness: {record['max'][0]}")
    print(f"  Failure Rate: {record['avg'][1]}")
    
    # Write to CSV
    with open('stats.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'generation': gen,
            'avg_fitness': record['avg'][0],
            'median_fitness': record['median'][0],
            'max_fitness': record['max'][0],
            'failure_rate': record['avg'][1],
            'time_elapsed': time.time() - start_time
        })

    # Save the model weights after each generation
    for i, model in enumerate(population):
        model.save_weights(f'model_weights_gen_{gen}_ind_{i}.h5')
