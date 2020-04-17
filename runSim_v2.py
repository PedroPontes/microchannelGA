import os
import numpy as np
import pandas as pd
import operator
import random
import math

# Implemented memoization to avoid repetition of simulations
fitness_cache = {}


class Channel:  # this class defines a microchannel by its geometry and outputs code blocks for openFoam

    """ This class defines a microchannel geometry given it's dimmensions: channelWidth, channelHeight,
    wallWidth, baseThick, topThick"""

    def __init__(self, dimensions, cooler_width, cooler_length):
        self.channelWidth = dimensions[0]  # entry variables are here defined
        self.channelHeight = dimensions[1]
        self.wallWidth = dimensions[2]
        self.baseThick = dimensions[3]
        self.topThick = dimensions[4]
        self.values = dimensions
        self.coolerWidth = cooler_width
        self.coolerLength = cooler_length

    def read(self):
        return "(" + str(self.channelWidth) + "," + str(self.channelHeight) + "," + str(self.wallWidth) + "," \
               + str(self.baseThick) + "," + str(self.topThick) + ")"

    def channel_count(self):
        return math.floor(self.coolerWidth / (self.channelWidth + self.wallWidth))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value
        self.channelWidth = self.values[0]  # entry variables are here redefined
        self.channelHeight = self.values[1]
        self.wallWidth = self.values[2]
        self.baseThick = self.values[3]
        self.topThick = self.values[4]

    def __iter__(self):
        return iter(self.values)

    def __repr__(self):
        return "(" + str(self.channelWidth) + "," + str(self.channelHeight) + "," + str(self.wallWidth) + "," \
               + str(self.baseThick) + "," + str(self.topThick) + ")"


# Some functions to help me process the files -------------------------------------------------------------------------
def read_mesh_file(path):  # to read the file
    f = open(path, 'r')
    contents = f.read()
    f.close()
    return contents


def write_mesh_file(path, content):  # to write stuff in the file
    f = open(path, 'w')
    f.write(content)
    f.close()


def create_blockMesh(channel, template_name, cell_size):  # this will create a blockMesh file string

    """ This method is used to create a blockMeshDict file according to the specific gemetric characteristics """

    # Defining some important variables
    xmax = (channel.channelWidth + channel.wallWidth) / 2
    ymax = channel.baseThick + channel.channelHeight + channel.topThick
    zmax = channel.coolerLength

    # There is a template prepared. This section cuts it, prepares the block of text with the geometry, and
    # returns block_mesh, a string with the entire text in the blockMeshDict
    text_block_mesh = read_mesh_file(template_name)
    text_block_mesh = text_block_mesh.split('// *split here*')
    geometry_block = 'xmax {:.3e}; \nymax {:.3e}; \nzmax {:.3e}; \n'.format(xmax, ymax, zmax)
    cell_block = 'xcells {:5d}; \nycells {:5d}; \nzcells {:5d}; \n'.format(round(xmax/cell_size),
                                                                           round(ymax/cell_size),
                                                                           50)
    block_mesh = text_block_mesh[0] + geometry_block + cell_block + text_block_mesh[1]
    return block_mesh


def create_topoSet(channel, template_name):  # this will create a topoSet file string

    """ This method is used to create a topoSetDict file according to the specific gemetric characteristics """

    # Defining some important variables
    xfluid = channel.channelWidth / 2
    xmax = (channel.channelWidth + channel.wallWidth) / 2
    yfluid = channel.baseThick
    ywall = yfluid + channel.channelHeight
    ymax = ywall + channel.topThick
    zmax = channel.coolerLength

    # There is a template prepared. This section cuts it, prepares the block of text with the geometry, and
    # returns block_mesh, a string with the entire text in the topoSetDict
    text_topo_set = read_mesh_file(template_name)
    text_topo_set = text_topo_set.split('// *split here*')
    geometry_block = ('xfluid {:.3e}; \nxmax {:.3e}; \nyfluid {:.3e}; \nywall {:.3e}; \
     \nymax {}; \nzmax {:.3e}; \n'.format(xfluid, xmax, yfluid, ywall, ymax, zmax))
    topo_set = text_topo_set[0] + geometry_block + text_topo_set[1]
    return topo_set


def launch_openfoam(blockmesh, toposet):
    os.system('rm -r runFolder')
    os.system('cp -r CaseFolder runFolder')
    write_mesh_file('runFolder/system/blockMeshDict', blockmesh)
    write_mesh_file('runFolder/system/topoSetDict', toposet)
    os.chdir('runFolder')
    os.system('foamCleanTutorials > fctlog')
    os.system('./makeMesh')
    os.system('chtSteady > log')
    os.system('cat log  | grep -A 2 \'Solving for solid region metal\' | awk \'{print $3}\' | sed -n \'3~4p\' > solidT')
    os.chdir('..')


def get_fitness():
    balance = read_mesh_file('runFolder/solidT')
    balance = balance.split("\n")
    balance = [float(x) for x in balance[:-1]]
    fitness = 1 / balance[-1]
    return fitness


# Ends here -----------------------------------------------------------------------------------------------------------


# These next functions serve the genetic algorithm --------------------------------------------------------------------
def generate_channels(precision, limit, cooler_width, cooler_length):  # precision and limit in m!
    dimensions = []
    for i in range(5):
        dimensions.append(round(random.randint(0, (limit[1] - limit[0])/ precision) * precision + limit[0], 5))
    channel = Channel(dimensions, cooler_width, cooler_length)
    return channel


def generate_population(pop_size, precision, limit, cooler_width, cooler_length):
    population = []
    for i in range(pop_size):
        population.append(generate_channels(precision, limit, cooler_width, cooler_length))
    return population


def rank_population(population, cell_size):
    fitness_results = {}

    for i in range(0, len(population)):
        print(str(i / len(population) * 100) + '%')
        if population[i].read() in fitness_cache:
            fitness_results[i] = fitness_cache[population[i].read()]
        else:
            blockmesh = create_blockMesh(population[i], 'blockMeshTemplate', cell_size)  # get the blockMesh files
            toposet = create_topoSet(population[i], 'topoSetTemplate')

            launch_openfoam(blockmesh, toposet)  # run OpenFoam
            fitness_results[i] = get_fitness()  # * population[i].channel_count()
            fitness_cache[population[i].read()] = fitness_results[i]

    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(ranked_pop, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(ranked_pop), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(ranked_pop[i][0])

    for j in range(0, len(ranked_pop) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(ranked_pop)):
            if pick <= df.iat[i, 3]:
                selection_results.append(ranked_pop[i][0])
                break

    return selection_results


def create_mating_pool(population, selection_results):
    mating_pool = []

    for i in range(len(selection_results)):
        index = selection_results[i]
        mating_pool.append(population[index])
    return mating_pool


def breed(parent1, parent2):
    child = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(len(parent1)):
        if start_gene <= i <= end_gene:
            child.append(parent1[i])
        else:
            child.append(parent2[i])

    return child


def breed_population(mating_pool, elite_size, cooler_width, cooler_length):
    children_population = []
    length = len(mating_pool) - elite_size
    sample_pool = random.sample(mating_pool, len(mating_pool))

    for i in range(0, elite_size):
        children_population.append(mating_pool[i])

    for i in range(0, length):
        child = Channel(breed(sample_pool[i], sample_pool[len(mating_pool) - i - 1]), cooler_width, cooler_length)
        children_population.append(child)
    return children_population


def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            gene_1 = individual[swapped]
            gene_2 = individual[swap_with]

            individual[swapped] = gene_1
            individual[swap_with] = gene_2
    return individual


def mutate_population(population, mutation_rate):
    mutated_pop = []

    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate, cooler_width, cooler_length, cell_size):
    population_ranked = rank_population(current_gen, cell_size)
    selection_results = selection(population_ranked, elite_size)
    mating_pool = create_mating_pool(current_gen, selection_results)
    new_gen = breed_population(mating_pool, elite_size, cooler_width, cooler_length)
    next_gen = mutate_population(new_gen, mutation_rate)
    return next_gen


def optimize_ga_script(pop_size, precision, limit, elite_size, mutation_rate, generations, cooler_width,
                       cooler_length, cell_size):
    """ This method is used to run the GA till the number of generations is achieved, using all the parameters"""
    next_gen = generate_population(pop_size, precision, limit, cooler_width, cooler_length)

    for i in range(0, generations):
        next_gen = next_generation(next_gen, elite_size, mutation_rate, cooler_width, cooler_length, cell_size)
        print(str(i + 1) + " out of " + str(generations) + " generations completed")

    best_geometry = next_gen[rank_population(next_gen)[0][0]]

    return best_geometry
# Ends here -----------------------------------------------------------------------------------------------------------


# print(optimize_ga_script(100, 2E-4, 1E-3, 5, 0.01, 100))

# pickle.dump(fitness_cache, open("save.p", "wb"))  # Save the cache
