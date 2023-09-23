__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import numpy as np

agentName = "random"

# This agent is not trainable
trainingSchedule = None

class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This agent doesn't evolve, and it doesn't have a chromosome.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

    def AgentFunction(self, percepts):
        # This agent ignores percepts and chooses a random action.  Your agents should not
        # perform random actions - your agents' actions should be deterministic from
        # computation based on self.chromosome and percepts
        return np.random.randint(low=-100,high=100,size=self.nActions)


