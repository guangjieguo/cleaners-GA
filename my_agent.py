__author__ = "Guangjie Guo"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "guo_guangjie@163.com"

import numpy as np

agentName = "<my_agent>"

generation_count = 0
fitness_5g = np.zeros((40))

# trainingSchedule = None
# trainingSchedule = [('random', 100), ('self', 100), ('random', 100), ('self', 100),
#                     ('random', 100)]
trainingSchedule = [('random', 10000)]

def top_n_indices(lst, n):
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:n]
    return indices

# This is the class for cleaner/agent
class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

        self.chromosome = np.random.uniform(0, 1, size=(self.nActions,
                                                int((self.nPercepts-14)+(self.nPercepts-14)*(self.nPercepts-15)/2)))

    def AgentFunction(self, percepts):

        # The percepts are a tuple consisting of four pieces of information
        #
        # visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
        #          is a 3x5x4 tensor, giving four maps with different information
        #          - the dirty,clean squares
        #          - the energy
        #          - the friendly and enemy cleaners that are able to traverse vertically
        #          - the friendly and enemy cleaners that are able to traverse horizontally
        #
        #  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
        #           the cleaner can still perform before it runs out of charge
        #
        #  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        #
        #  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
        #          backward movement might fail if it would result in a collision with another robot); fails=0 means
        #          the last action succeeded.


        visual, energy, bin, fails = percepts

        # You can further break down the visual information

        floor_state = visual[:,:,0]   # 3x5 map where -1 indicates dirty square, 0 clean one
        energy_locations = visual[:,:,1] #3x5 map where 1 indicates the location of energy station, 0 otherwise
        vertical_bots = visual[:,:,2] # 3x5 map of bots that can in this turn move up or down (from this bot's point of
                                      # view), -1 if the bot is an enemy, 1 if it is friendly
        horizontal_bots = visual[:,:,3] # 3x5 map of bots that can in this turn move left or right (from this bot's point
                                        # of view), -1 if the bot is an enemy, 1 if it is friendly

        #You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        # a mape where -1 indicates dirty square, 0 a clean one, and 1 an energy station.

        combined = floor_state + energy_locations

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.
        #
        # The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

        # The index of the largest value in the 'actions' vector/list is the action to be taken,
        # with the following interpretation:
        # largest value at index 0 - move forward;
        # largest value at index 1 - turn right;
        # largest value at index 2 - turn left;
        # largest value at index 3 - move backwards;
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        # My agent is using quadratic polynomial as agent function.
        x_vector = np.hstack((combined.flatten(), vertical_bots.flatten(), horizontal_bots.flatten()))
        x_vector = x_vector.tolist()
        x_vector.append(energy)
        x_vector.append(bin)
        x_vector.append(fails)
        for i in range(self.nPercepts-15):
            for j in range(self.nPercepts-15):
                if j >= i:
                    x_vector.append(x_vector[i]*x_vector[j])
        x_vector.append(1)
        x_vector = np.array(x_vector)
        action_vector = np.dot(self.chromosome, x_vector)
        return action_vector

def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, cleaner in enumerate(population):
        # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
        # game engine, which is a dictionary with the following information on the performance of the cleaner in
        # the last game:
        #
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        # This fitness functions considers total number of cleaned squares.  This may NOT be the best fitness function.
        # You SHOULD consider augmenting it with information from other stats as well.  You DON'T HAVE TO make use
        # of every stat.

        # This is the best one among many cases I have tried.
        fitness[n] = cleaner.game_stats['emptied'] + cleaner.game_stats['visits']/15

    return fitness


def newGeneration(old_population):

    # This function should return a tuple consisting of:
    # - a list of the new_population of cleaners that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    # Fetch the game parameters stored in each agent (we will need them to
    # create a new child agent)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns


    fitness = evalFitness(old_population)
    global generation_count, fitness_5g
    generation_count += 1
    fitness_5g += fitness
    # Create new population list...
    if generation_count%5 == 0:
        numelits = 15
        mutation_rate_child = 1
        mutation_rate = 0.01
        new_population = list()
        indices = top_n_indices(fitness_5g, numelits)
        for index in indices:
            new_population.append(old_population[index])
        probabilities = fitness_5g/np.sum(fitness_5g)
        for n in range(N-numelits):

            # Create a new cleaner
            new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

            # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
            # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome
            par1Index, par2Index = np.random.choice(range(N), replace=False, p=probabilities, size=2)
            par1 = old_population[par1Index]
            par2 = old_population[par2Index]
            # cutmiddle = int(((nPercepts - 14) + (-14 + nPercepts) * (nPercepts-15)/2)/2)
            # new_cleaner.chromosome[:, :cutmiddle] = par1.chromosome[:, :cutmiddle]
            # new_cleaner.chromosome[:, cutmiddle:] = par2.chromosome[:, cutmiddle:]
            gamma = np.random.uniform(0, 1)
            new_cleaner.chromosome = gamma*par1.chromosome + (1-gamma)*par2.chromosome


            # Consider implementing elitism, mutation and various other
            # strategies for producing a new creature.
            random_float = np.random.uniform(0, 1)
            if random_float < mutation_rate_child:
                random_matrix = np.random.random((nActions, int((nPercepts-14) + (-14+nPercepts) * (nPercepts-15) / 2)))
                for i in range(nActions):
                    for j in range(int((nPercepts - 14) + (-14 + nPercepts) * (nPercepts-15) / 2)):
                        if random_matrix[i, j] < mutation_rate:
                            # new_cleaner.chromosome[i, j] = np.random.normal(0, 1)
                            new_cleaner.chromosome[i, j] = np.random.uniform(0, 1)
                            # new_cleaner.chromosome[i, j] = np.random.randint(-5, 6)
                            # new_cleaner.chromosome[i, j] = np.random.randint(0, 101)

            # Add the new cleaner to the new population
            new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
        avg_fitness = np.mean(fitness)
        fitness_5g = np.zeros((40))

        return (new_population, avg_fitness)
    else:
        new_population = old_population
        return (new_population, np.mean(fitness))
