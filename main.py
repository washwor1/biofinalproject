import random
import time
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from keras.models import clone_model
import playgame as pg
import model as m
import tensorflow as tf

# Variables for GA
POP_SIZE = 10
NGEN = 10
CXPB = 0.7
MUTPB = 0.3

# Define the fitness function
def fitness(individual):
    weights = array_to_weights(individual)
    model = clone_model(base_model)
    model.set_weights(weights)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    fit, _ = pg.play_game(model)
    return fit,

# Transform array of weights to the weight and bias shapes for the layers in the model

import numpy as np

def array_to_weights(weight_values):
    weight_shapes = [(3, 3, 4, 16), (16,), (3, 3, 16, 32), (32,), (3, 3, 32, 64), (64,),
                     (203520, 96), (24, 96), (96,), (24, 15), (15,)]
    
    weights = []
    start = 0
    
    for shape in weight_shapes:
        size = np.prod(shape)
        weights.append(np.array(weight_values[start:start+size]).reshape(shape))
        start += size
    
    return weights




# Define the mutation operation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTPB:
            individual[i] += random.gauss(0, 0.1)
    return individual,

# Create the base model
input_shape = (880, 1000, 4)
base_model = m.create_cnn_lstm_model(input_shape)

# bruh =base_model.get_weights()
# for array in bruh:
#     print(array.shape)

print(base_model.layers[8].output_shape)

# Initialize the DEAP tools
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
print("a\n")
toolbox = base.Toolbox()
print("a\n")

def attr_float():
    return random.uniform(-1, 1)

toolbox.register("attr_float", attr_float)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, sum([np.prod(w.shape) for w in base_model.get_weights()]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
print("a\n")

# Initialize the population and hall of fame
pop = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)

# Track statistics
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("std", np.std)

logbook = tools.Logbook()  # create logbook outside the loop
times = []

for gen in range(NGEN):
    start_time = time.time()
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=1, stats=stats, halloffame=hof, verbose=True)
    end_time = time.time()
    times.append(end_time - start_time)
    logbook.record(time_per_gen=times[-1], **log[0])  # update the logbook with the new log

# Save the statistics to a CSV
df = pd.DataFrame(logbook)
df['time_per_gen'] = times
df.to_csv('evolution_stats.csv', index=False)

# Output the best individual and its fitness score
best_individual = hof[0]
best_fitness = fitness(best_individual)
print(f"Best Individual: {best_individual}")
print(f"Best Fitness: {best_fitness}")

# Save the weights of the best individual
best_weights = array_to_weights(best_individual)
np.save('best_weights.npy', best_weights)
