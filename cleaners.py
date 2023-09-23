__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__version__ = 1.1

import importlib
import numpy as np
import traceback
import sys
import gzip, pickle
from datetime import datetime
import os
import signal
import time

import matplotlib.pyplot as plt

maxTrainingEpochs = 10000
maxActions = 4
numPlays = 5
fieldOfVision = 5
maxBin = 3


def alarm_handler(signum, frame):
    raise RuntimeError("Time out")

def percepts_global_to_agent_frame_of_reference(percepts,rotation):

    if rotation == 90:
        percepts = np.rot90(percepts, axes=[0, 1])
    elif rotation == 270:
        percepts = np.rot90(percepts, axes=[1, 0])
    elif rotation == 180:
        percepts = np.rot90(np.rot90(percepts,axes=[0,1]),axes=[0,1])


    return percepts

def actions_agent_to_global_shift(rotation):

    if rotation == 0:
        return -1,0
    elif rotation == 90:
        return 0,1
    elif rotation == 180:
        return 1,0
    elif rotation == 270:
        return 0,-1
    else:
        raise RuntimeError("Rotation %d, should be either 0,90,180 or 270.")

def check_coordinate_wraparound(y,x,Y,X):

    if y < 0:
        y += Y
    elif y >= Y:
        y -= Y

    if x < 0:
        x += X
    elif x >= X:
        x -= X

    return y,x


def  move_on_square_in_direction(y,x,direction,Y,X):

    yd, xd = actions_agent_to_global_shift(direction)

    y += yd
    x += xd

    return check_coordinate_wraparound(y,x,Y,X)


# Class avatar is a wrapper for the agent with extra bits required
# for running the game
class Avatar:

    # Initialise avatar for an agent of a given player
    def __init__(self,agent,player):
        self.agent = agent
        self.player = player


    # Reset the avatar variables for a new game
    def reset_for_new_game(self,position,rotation,gridSize):
        self.energy = self.player.game.maxEnergy
        self.position = position
        self.rotation = rotation
        self.start_position = position
        self.bin = 0
        self.emptied = 0
        self.cleaned = 0
        self.active_turns = 0
        self.successful_actions = 0
        self.recharge_count = 0
        self.recharge_energy = 0
        self.success = 1
        self.fails = 0
        self.map = np.zeros(gridSize).astype('uint8')

    # Execute AgentFunction that maps percepts to actions
    def action(self, turn, percepts):

        if self.player.game.in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(1)
        try:
            action = self.agent.AgentFunction(percepts)

        except Exception as e:
            self.player.game.throwError(str(e))
            return

        if self.player.game.in_tournament:
            signal.alarm(0)

        if type(action) != list and type(action) != np.ndarray:
            self.player.game.throwError("AgentFunction must return a list or numpy array")
            return

        if len(action) != maxActions:
            self.player.game.throwError("The returned action list/array is of length %d, it must be of length %" % (len(action),maxActions))
            return

        return action

# Class player holds all the agents for a given player
class Player:

    def __init__(self, game, player, playerFile,emptyMode=False,jointname=False):

        self.game = game
        self.player = player
        self.playerFile = playerFile
        self.nAgents = self.game.nAgents
        self.fitness = list()
        self.errorMsg = ""
        self.ready = False

        if emptyMode:
            return

        if type(playerFile) == Player:
            playerObj = playerFile
            self.playerFile = playerObj.playerFile
            self.exec = playerObj.exec
            self.name = playerObj.name
            self.trainingSchedule = []
            self.trained = True

            try:
                self.agents_to_avatars(playerObj.agents)
            except Exception as e:
                self.game.throwError(str(e), self.player)
                return

        else:
            if not os.path.exists(playerFile):
                self.game.throwError("Error! Agent file '%s' not found" % self.playerFile, playerid)
                return

            if len(playerFile) > 3 and playerFile[-3:].lower() == '.py':
                playerModule = playerFile[:-3]
            else:
                self.game.throwError("Error! Agent file %s needs a '.py' extension" % self.playerFile, playerid)
                return

            # Import agent file as module
            if self.game.in_tournament:
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(10)
            try:
                if self.game.in_tournament and playerModule != 'random_agent':
                    self.exec = importlib.machinery.SourceFileLoader('my_agent', playerModule + '.py').load_module()
                else:
                    self.exec = importlib.import_module(playerModule)

            except Exception as e:
                self.throwError(str(e),self.player)
                return

            if self.game.in_tournament:
                signal.alarm(0)

            if hasattr(self.exec, 'agentName') and self.exec.agentName[0] != '<':
                self.name = self.exec.agentName
            else:
                if self.game.in_tournament and playerFile != 'random_agent.py':
                    self.name = playerFile.split('/')[-2]# playerFile.split('.')[1]
                else:
                    self.name = playerFile

            if jointname and self.game.in_tournament:
               self.pname = playerFile.split('/')[-2]

            if not hasattr(self.exec,'trainingSchedule'):
                self.game.throwError("Agent is missing the 'trainingSchedule' variable.",self.player)
                return

            self.trainingSchedule = self.exec.trainingSchedule

            if self.trainingSchedule is not None and not isinstance(self.trainingSchedule,list):
                self.game.throwError("Error! Agent's 'trainingSchedule' should be a list of (str,int) tuples.",self.player)
                return

            if isinstance(self.trainingSchedule, list):

                totTrainEpochs = 0

                for trainSession in self.trainingSchedule:
                    if not isinstance(trainSession,tuple) or len(trainSession) < 2 or not (isinstance(trainSession[0],str)) or not isinstance(trainSession[1],int):
                        self.game.throwError("Agent's 'trainingSchedule' should be a list containing (str,int) tuples.",self.player)
                        return

                    if trainSession[1] < 0:
                        self.game.throwError("Agent's 'trainingSchedule' should be a list of (str,int) tuples, where int corresponds to the number of train generations.",selfp.playerid)
                        return

                    totTrainEpochs += trainSession[1]

                if self.game.in_tournament and totTrainEpochs > maxTrainingEpochs:
                    self.game.throwError("Agent's 'trainingSchedule' cannot specify more than %d training epochs in total." % maxTrainingEpochs)
                    return

            if self.trainingSchedule is None or self.game.training == 'none':
                self.trained = True
            else:
                self.trained = False

            # Create the initial population of agents by creating
            # new instance of the agent using provided MyCreature class
            agentFile = playerModule

            if self.game.in_tournament and agentFile != 'random_agent':
                if self.game.training == 'pretrained':
                    agentFileSave = "/".join(agentFile.split('/')[:-1] + ['my_agent'])
                else:
                    agentFileSave = "/".join(agentFile.split('/')[:-1] + [agentFile.split('/')[-2]])
            else:
                agentFileSave = agentFile

            self.savedAgent = agentFileSave + '.tar.gz'

            savedAgent = self.savedAgent

            agents = []
            if self.game.training == 'none' or not os.path.exists(savedAgent) or (not self.game.in_tournament and os.path.getmtime(savedAgent) < os.path.getmtime("%s.py" % agentFile)):
                pass
            else:
                with gzip.open(savedAgent,'r') as f:
                    agents_saved = pickle.load(f)
                if self.game.nAgents == len(agents_saved):
                    agents = agents_saved
                    self.trained = True


            if len(agents) == 0:
                agents = list()
                for n in range(self.nAgents):
                    if self.game.in_tournament:
                        signal.signal(signal.SIGALRM, alarm_handler)
                        signal.alarm(1)
                    try:
                        agent = self.exec.Cleaner(nPercepts=fieldOfVision*(fieldOfVision//2+1)*4+3, nActions=maxActions, gridSize=self.game.gridSize, maxTurns=self.game.nTurns)
                    except Exception as e:
                        self.game.throwError(str(e),self.player)
                        return

                    if self.game.in_tournament:
                        signal.alarm(0)
                    agents.append(agent)


            # Convert list of agents to list of avatars
            try:
                self.agents_to_avatars(agents)
                self.agents = agents
            except Exception as e:
                self.game.throwError(str(e),self.player)
                return

        self.ready = True

    # Convert list of agents to list of avatars
    def agents_to_avatars(self, agents):
        self.avatars = list()
        self.stats = list()

        for agent in agents:
            if type(agent) != self.exec.Cleaner:
                if self.game.in_tournament:
                    raise RuntimeError(
                        'The new_population returned from newGeneration() must contain objects of Cleaner() type')
                else:
                    print("The new_population returned form newGeneration() in '%s' must contain objects of Cleaner() type" %
                    self.playerFile)
                    traceback.print_exc()
                    sys.exit(-1)

            avatar = Avatar(agent,player=self)
            self.avatars.append(avatar)
            self.stats.append(dict())

    def avatar_to_agent_stats(self,avatar):
        agent = avatar.agent
        agent.game_stats = {}
        agent.game_stats['emptied'] = avatar.emptied
        agent.game_stats['cleaned'] = avatar.cleaned
        agent.game_stats['active_turns'] = avatar.active_turns
        agent.game_stats['successful_actions'] = avatar.successful_actions
        agent.game_stats['recharge_count'] = avatar.recharge_count
        agent.game_stats['recharge_energy'] = avatar.recharge_energy
        agent.game_stats['visits'] = np.sum(avatar.map)
        return agent

    # Get a new generation of agents
    def new_generation_agents(self,gen):

        # Record game stats in the agent objects
        old_population = list()
        for avatar in self.avatars:
            agent = self.avatar_to_agent_stats(avatar)
            old_population.append(agent)

        if self.playerFile != 'random_agent.py':
            msg = "  avg_fitness: "

            if self.game.in_tournament:
                self.game.train_report.append(msg)

            if self.game.verbose:
                sys.stdout.write(msg)
                sys.stdout.flush()

        # Get a new population of agents by calling
        # the provided newGeneration method
        if self.game.in_tournament:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(4)

        try:
            result = self.exec.newGeneration(old_population)
        except Exception as e:
            if self.game.in_tournament:
                raise RuntimeError('Failed to execute newGeneration(), %s' % str(e))
            else:
                print("Failed to execute newGeneration() from '%s', %s" % (self.playerFile, str(e)))
                traceback.print_exc()
                sys.exit(-1)

        if self.game.in_tournament:
            signal.alarm(0)

        if type(result) != tuple or len(result) != 2:
            if self.game.in_tournament:
                raise RuntimeError('The returned value form newGeneration() must be a 2-item tuple')
            else:
                print("The returned value form newGeneration() in '%s.py' must be a 2-item tuple" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        (new_population, fitness) = result

        ##########
        avg_fitnesses.append(fitness)

        if type(new_population) != list:
            if self.game.in_tournament:
                raise RuntimeError('The new_population returned form newGeneration() must be a list')
            else:
                print("The new_population returned form newGeneration() in '%s.py' must be a list" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        try:
            fitness = float(fitness)
        except Exception as e:
            if self.game.in_tournament:
                raise RuntimeError('The fitness returned form newGeneration() must be float or int')
            else:
                print("The new_population returned form newGeneration() in '%s.py' must be a float or int" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(new_population) != len(old_population):
            if self.game.in_tournament:
                raise RuntimeError('The new_population returned form newGeneration() must contain %d items' % self.nAgents)
            else:
                print("The new_population returned form newGeneration() in '%s.py' must contain %d items" % (self.playerFile, self.nAgents))
                traceback.print_exc()
                sys.exit(-1)

        if self.playerFile != 'random_agent.py':
            msg = " %.2e" % fitness
            if self.game.in_tournament:
                self.game.train_report.append(msg)

            if self.game.verbose:
                sys.stdout.write(msg)
                sys.stdout.flush()

        self.fitness.append(fitness)

        # Convert agents to avatars
        self.agents_to_avatars(new_population)

    def evaluate_fitness(self):

        agents = []
        for avatar in self.avatars:
            agent = self.avatar_to_agent_stats(avatar)
            agents.append(agent)

        try:
            fitness = self.exec.evalFitness(agents)
        except:
            if self.game.in_tournament:
                raise RuntimeError("Failed to execute evalFitness() from '%s'" % self.playerFile)
            else:
                print("Failed to execute evalFitness() from '%s'" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if isinstance(fitness,np.ndarray):
            fitness = fitness.tolist()

        if not isinstance(fitness, list):
            if self.game.in_tournament:
                raise RuntimeError("Function evalFitness() from '%s' must return a list" % self.playerFile)
            else:
                print("Function evalFitness() from '%s' must return a list" % self.playerFile)
                traceback.print_exc()
                sys.exit(-1)

        if len(fitness) != len(agents):
            if self.game.in_tournament:
                raise RuntimeError(
                    "Length of the list returned by evalFitness() from '%s' is %d; expecting the length to be %d." % (
                    self.playerFile, len(fitness), len(agents)))
            else:
                print(
                    "Length of the list returned by evalFitness() from '%s' is %d; expecting the length to be %d." % (
                    self.playerFile, len(fitness), len(agents)))
                traceback.print_exc()
                sys.exit(-1)

        self.fitness.append(np.mean(fitness))
        I = np.argsort(fitness)[::-1]
        self.avatars = np.array(self.avatars)[I].tolist()
        msg = "  avg_fitness:  %.2e\n\n" % np.mean(fitness)

        ######
        avg_fitnesses.append(np.mean(fitness))

        if self.game.in_tournament:
            self.game.train_report.append(msg)

        if self.game.verbose:
            sys.stdout.write(msg)
            sys.stdout.flush()


    def save_trained(self):

        savedAgent = self.savedAgent

        if self.game.verbose:
            sys.stdout.write("Saving last generation agents to %s..."  % self.savedAgent)
            sys.stdout.flush()
        agents = []
        for avatar in self.avatars:
            agents.append(avatar.agent)

        with gzip.open(savedAgent,'w') as f:
            pickle.dump(agents, f)

        if self.game.verbose:
            sys.stdout.write("done\n")
            sys.stdout.flush()


class CleanersPlay:

    def __init__(self,game,showGame=None,saveGame=False):
        self.game = game

        Y, X = self.game.gridSize
        self.map = np.zeros((Y, X, 2), dtype='int8')
        self.map[:,:,0] = -1
        self.rotations = [0,90,180,270]

        self.showGame = showGame
        self.saveGame = saveGame


    def vis_update(self,players):

        stats = {}
        stats['cleaned'] = [0]*len(players)
        cleaners = []

        for p,player in enumerate(players):
            for z,avatar in enumerate(player.avatars):
                stats['cleaned'][p] += avatar.cleaned
                y,x = avatar.position
                r = avatar.rotation
                e = avatar.energy/player.game.maxEnergy
                b = avatar.bin/maxBin
                cleaners.append((y,x,r,e,b,p))

        return cleaners, stats

    def manhattan_distance(self, x1,y1,x2,y2):
        x = np.min([np.abs(x1-x2),np.abs(x2-x1)])
        y = np.min([np.abs(y1-y2),np.abs(y2-y1)])

        return x+y

    def play(self,players):

        Y,X,_ = np.shape(self.map)

        tiles = []


        for y in range(Y):
            for x in range(X):
                if self.map[y,x,0] != 10:
                    tiles.append((y,x))

        I = self.game.rnd_fixed_seed.permutation(len(tiles))
        tiles = np.array(tiles)[I]

        player1 = players[0]

        for id, avatar in enumerate(player1.avatars):

            while True:
                y,x = tiles[-1]
                tiles = tiles[:-1]

                if self.map[y,x,1] != 0:
                    continue

                yf = Y - y - 1
                xf = X - x - 1

                if yf == y and xf == x:
                    # Can't have cleaners in the same spot
                    continue

                rotation = self.game.rnd_fixed_seed.choice(self.rotations)

                avatar.reset_for_new_game((y,x),rotation,(Y,X))
                self.map[y,x,1] = rotation/90+1
                self.map[y,x,0] = 1
                for i in [-1,0,1]:
                    yo = y+i
                    if yo < 0:
                        yo += Y
                    elif yo >= Y:
                        yo -= Y
                    for j in [-1,0,1]:
                        xo = x+j
                        if xo < 0:
                            xo += X
                        elif xo >= X:
                            xo -= X
                        if self.map[yo,xo,0] == -1:
                            self.map[yo,xo,0] = 0

                if len(players) > 1:

                    rotation = rotation + 180
                    rotation %= 360

                    avatar = players[1].avatars[id]
                    avatar.reset_for_new_game((yf,xf),rotation,(Y,X))
                    self.map[yf,xf,1] = -(rotation/90+1)
                    self.map[yf,xf,0] = 1
                    for i in [-1, 0, 1]:
                        yo = yf + i
                        if yo < 0:
                            yo += Y
                        elif yo >= Y:
                            yo -= Y
                        for j in [-1, 0, 1]:
                            xo = xf + j
                            if xo < 0:
                                xo += X
                            elif xo >= X:
                                xo -= X
                            if self.map[yo, xo,0] == -1:
                                self.map[yo, xo,0] = 0
                break

        if self.showGame is not None or self.saveGame:
            vis_cleaners, stats = self.vis_update(players)
            vis_data = (self.map[:,:,0], vis_cleaners, stats)

            if self.showGame is not None:
                self.game.vis.show(vis_data, turn=0, titleStr=self.showGame)

            if self.saveGame:
                self.vis_data = [vis_data]

        all_avatars = []
        for player in players:
            all_avatars += player.avatars

        # Play the game over a number of turns
        for turn in range(self.game.nTurns):
            gameDone = True

            self.map[:, :, 1] = 0
            for k, player in enumerate(players):

                if k == 0:
                    kj = 1
                else:
                    kj = -1

                for avatar in player.avatars:
                    y,x = avatar.position
                    self.map[y,x,1] = kj*(avatar.rotation/90+1)

            # Get actions of the agents
            # Reset avatars for a new game
            for k, player in enumerate(players):

                for avatar in player.avatars:

                    if avatar.energy < 1:
                        continue

                    # Percepts
                    percepts = np.zeros((fieldOfVision,fieldOfVision,2)).astype('int')

                    pBHalf = fieldOfVision // 2

                    # Add nearby agents to percepts

                    for i,io in enumerate(range(-pBHalf,pBHalf+1)):
                        for j,jo in enumerate(range(-pBHalf,pBHalf+1)):
                            y, x = avatar.position
                            y += io
                            x += jo

                            y,x = check_coordinate_wraparound(y,x,Y,X)

                            percepts[i,j,:] = self.map[y,x,:]
                    if k==0:
                        jk=1
                    else:
                        jk=-1

                    percepts[:,:,1] *= jk
                    y,x = avatar.position

                    if avatar.energy > 0:
                        gameDone = False

                    percepts= percepts_global_to_agent_frame_of_reference(percepts,avatar.rotation)
                    ri = int(avatar.rotation/90)
                    percepts[:,:,1] = np.sign(percepts[:,:,1])*(np.abs(np.abs(percepts[:,:,1])-ri))

                    percepts_sparse = np.zeros((fieldOfVision,fieldOfVision,6)).astype('int')
                    pfield = np.copy(percepts[:, :, 0])
                    pfield[pfield == 1] = 0
                    percepts_sparse[:,:,0] = pfield
                    pfield = np.copy(percepts[:, :, 0])
                    pfield[pfield < 0] = 0
                    percepts_sparse[:,:,1] = pfield
                    for i in range(1,5):
                        pver = np.copy(percepts[:, :, 1])
                        apver = np.abs(pver)
                        zpver = np.zeros(np.shape(pver),dtype=pver.dtype)
                        zpver[apver == i] = np.sign(pver[apver == i])
                        percepts_sparse[:,:,i+1] = zpver

                    percepts_sparse = percepts_sparse[:pBHalf+1]
                    percepts_sparse[:,:,2] += percepts_sparse[:,:,4]
                    percepts_sparse[:,:,3] += percepts_sparse[:,:,5]

                    percepts_sparse = percepts_sparse[:,:,:4]

                    percepts = (percepts_sparse, avatar.energy, maxBin - avatar.bin, avatar.fails)

                    # Get action from agent
                    try:
                        action = avatar.action(turn+1,percepts)
                    except Exception as e:
                        if self.game.in_tournament:
                            self.game.game_scores[k] += [-500]
                            self.game.game_messages[k] = str(e)
                            self.game.game_play = False
                        else:
                            traceback.print_exc()
                            sys.exit(-1)

                    if not self.game.game_play:
                        break

                    ai = np.argmax(action)

                    avatar.new_position = avatar.position
                    if ai == 0 or ai==3:
                        if ai==3:
                            y, x = move_on_square_in_direction(y, x, (avatar.rotation+180)%360, Y, X)
                        else:
                            y, x = move_on_square_in_direction(y, x, avatar.rotation, Y, X)

                        avatar.new_position = (y,x)
                    elif ai == 1:
                        #Turn right
                        avatar.rotation += 90
                        avatar.rotation %= 360
                    elif ai == 2:
                        # Turn left
                        avatar.rotation -= 90
                        avatar.rotation %= 360

                    avatar.energy -= 1
                    avatar.action_success = 1

                    if avatar.energy < 0:
                        avatar.energy = 0

            collisions = True
            while collisions:
                collisions = False

                for i in range(len(all_avatars)):
                    avatar_i = all_avatars[i]
                    for j in range(i+1,len(all_avatars)):
                        avatar_j = all_avatars[j]

                        if avatar_i.new_position == avatar_j.new_position:
                            avatar_i.new_position = avatar_i.position
                            avatar_j.new_position = avatar_j.position
                            avatar_i.action_success = 0
                            avatar_j.action_success = 0
                            collisions = True
                        elif avatar_i.new_position == avatar_j.position and avatar_j.new_position == avatar_i.position:
                            #Can't swap
                            avatar_i.new_position = avatar_i.position
                            avatar_j.new_position = avatar_j.position
                            avatar_i.action_success = 0
                            avatar_j.action_success = 0
                            collisions = True

            # Check for energy renewed
            for avatar in all_avatars:
                if avatar.action_success == 1:
                    avatar.successful_actions += 1
                    avatar.fails = 0
                else:
                    avatar.fails += 1

                y, x = avatar.new_position
                avatar.map[y,x] = 1
                if self.map[y,x,0] == 1:
                    avatar.recharge_energy = self.game.maxEnergy-avatar.energy
                    avatar.recharge_count += 1
                    avatar.energy = self.game.maxEnergy
                    avatar.emptied += avatar.bin
                    avatar.bin = 0
                elif self.map[y,x,0] == -1:
                    if avatar.bin < maxBin:
                        avatar.bin += 1
                        avatar.cleaned += 1
                        self.map[y,x,0] = 0
                avatar.position = (y,x)
                if avatar.energy >= 1:
                    avatar.active_turns += 1

            if not self.game.game_play:
                return None

            if self.showGame is not None or self.saveGame:

                vis_cleaners, stats = self.vis_update(players)
                vis_data = (self.map[:,:,0], vis_cleaners, stats)

                if self.showGame is not None:
                    self.game.vis.show(vis_data, turn=turn + 1, titleStr=self.showGame)

                if self.saveGame:
                    self.vis_data.append(vis_data)

            if gameDone:
                break

            self.turn = turn

        if self.saveGame:
            if self.game.in_tournament:
                savePath = "/".join(self.game.players[0].playerFile.split('/')[:-1])
            else:
                savePath = "saved"
            if not os.path.isdir(savePath):
                os.makedirs(savePath, exist_ok=True)

            now = datetime.now()
            # Month abbreviation, day and year
            saveStr = now.strftime("%b-%d-%Y-%H-%M-%S")
            if len(players) == 1:
                saveStr += "-%s" % (players[0].name)
                name2 = None
            else:
                saveStr += "-%s-vs-%s" % (players[0].name, players[1].name)
                name2 = players[1].name

            if self.game.in_tournament:
                saveStr += '_%s' % self.game.training

            saveStr += ".pickle.gz"

            saveFile = os.path.join(savePath, saveStr)

            self.game.game_saves.append(saveFile)

            with gzip.open(saveFile, 'w') as f:
                pickle.dump((players[0].name, name2, self.vis_data, (Y,X)), f)

        scores = []
        for k, player in enumerate(players):
            scores.append(0)
            for avatar in player.avatars:
                scores[-1] += avatar.cleaned

        scores = np.array(scores).astype('int32')

        if len(scores) == 1:
            return scores[0]
        else:
            return scores[0]-scores[1]



# Class that runs the entire game
class CleanersGame:

    # Initialises the game
    def __init__(self, gridSize, nTurns, nChargers, nAgents, saveFinalGames=True,seed=None, tournament=False, verbose=True, training='trained'):

        self.rnd = np.random.RandomState()
        self.gridSize = gridSize
        self.nTurns = nTurns
        self.nActions = 5
        self.game_play = True
        self.in_tournament = tournament
        self.nChargers = nChargers
        self.nAgents = nAgents
        self.saveFinalGames = saveFinalGames
        self.rnd_fixed_seed = np.random.RandomState(seed)
        self.verbose = verbose
        self.training = training
        self.maxEnergy = np.min(gridSize)//2

        if self.in_tournament:
            self.throwError = self.errorAndReturn
            self.train_report = []
            self.game_report = []
        else:
            self.throwError = self.errorAndExit

    def errorAndExit(self, errorStr, playerid=None):
        raise RuntimeError(errorStr)

    def errorAndReturn(self, errorStr,playerid=None):
        signal.alarm(0)
        self.errorStr = errorStr
        return None


    # Run the game
    def run(self,player1File, player2File,visResolution=(720,480), visSpeed='normal',savePath="saved",
            trainers=[("random_agent.py","random")],runs = [1,2,3,4,5], shows = [1,2,3,4,5],jointname=False):

        self.players = list()

        self.game_messages = ['', '']
        self.game_scores = [[],[]]
        self.game_saves = list()

        # Load player 1
        if player1File is not None:
            try:
                self.players.append(Player(self,len(self.players),player1File,jointname=jointname))
            except Exception as e:
                if self.in_tournament:
                    self.players.append(Player(self,0,player1File,self.nAgents,emptyMode=True))
                    self.game_messages[0] = "Error! Failed to create a player with the provided code"
                else:
                    traceback.print_exc()
                    return

            if not self.players[0].ready:
                self.game_scores[0].append(-500)
                if self.players[0].errorMsg != "":
                    self.game_messages[0] = self.players[0].errorMsg

                self.game_play = False
            elif not self.players[0].trained:
                self.players[0] = self.train(self.players[0],visResolution,visSpeed)
                if self.players[0] is None:
                    self.game_scores[0].append(-500)
                    self.game_play = False

            # Load player 2
        if player2File is not None:
            try:
                self.players.append(Player(self,len(self.players),player2File,jointname=jointname))
            except Exception as e:
                if self.in_tournament:
                    self.players.append(Player(self,1,player2File,emptyMode=True))
                    self.game_messages[1] = "Error! Failed to create a player with the provided MyAgent.py code"
                else:
                    print('Error! ' + str(e))
                    sys.exit(-1)

            if not self.players[1].ready:
                self.game_scores[1].append(-500)
                if self.players[1].errorMsg != "":
                    self.game_messages[1] = self.players[1].errorMsg

                self.game_play = False
            elif not self.players[1].trained:
                self.players[1] = self.train(self.players[1],visResolution,visSpeed)
                if self.players[1] is None:
                    self.game_scores[1].append(-500)
                    self.game_play = False

        if not self.game_play:
            return

        if self.saveFinalGames:
            saves = shows
        else:
            saves = []

        self.play(self.players,runs,shows,saves,visResolution,visSpeed,savePath)


    def train(self,player,visResolution=(720,480), visSpeed='normal'):

        playerNumber = player.player
        trainingSchedule = player.trainingSchedule

        tot_gens = 0
        for op, gens in trainingSchedule:
            tot_gens += gens


        if tot_gens > maxTrainingEpochs:
            tot_gens = maxTrainingEpochs

        gens_count = 0

        for op, gens in trainingSchedule:

            if gens_count + gens > tot_gens:
                gens = tot_gens - gens_count

            if gens==0:
                break

            if op == 'random':
                opFile = 'random_agent.py'
            elif op == 'self':
                opFile = None
            else:
                opFile = op

            opponentNumber = (player.player + 1) % 2

            # Load opponent
            players = [player]

            if op == 'self':
                msg = "\nTraining %s against self for %d generations...\n" % (player.name, gens)
                #if self.in_tournament:
                #    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)

                opponent = Player(self, opponentNumber, player)
                players.append(opponent)
            else:
                try:
                    opponent = Player(self, opponentNumber, playerFile=opFile)
                except Exception as e:
                    if self.in_tournament:
                        self.game_messages[playerNumber] = "Error! Failed to create opponent '%s' in training" % op
                        return None
                    else:
                        traceback.print_exc()
                        sys.exit(-1)

                if not opponent.ready:
                    self.game_scores[player.player].append(-500)
                    if player.errorMsg != "":
                        self.game_messages[player.player] = player.errorMsg
                    return None

                msg = "\nTraining %s against %s for %d generations...\n" % (player.name, op, gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)
                players.append(opponent)
            #else:
            #    msg = "\nTraining %s in single-player mode for %d generations...\n" % (player.name, gens)
            #    if self.in_tournament:
            #        self.train_report.append(msg)
            #    if self.verbose:
            #        sys.stdout.write(msg)
            msg = "------"
            if self.in_tournament:
                self.train_report.append(msg)

            if self.verbose:
                sys.stdout.write(msg)


            self.play(players,[], [], [], visResolution, visSpeed, trainGames=(gens,gens_count,tot_gens))

            if not self.game_play:
                return None

            gens_count += gens
            if gens_count >= tot_gens:
                break

        try:
            player.save_trained()
        except Exception as e:
            if self.in_tournament:
                self.game_messages[playerNumber] = "Error! Failed to save training results."
                return None
            else:
                traceback.print_exc()
                sys.exit(-1)

        return player

    def play(self,players, run_games, show_games, save_games, visResolution=(720,480), visSpeed='normal',savePath="saved",trainGames=None):

        if len(show_games)>0:
            import vis_pygame as vis
            playerStrings = []
            for p in players:
                playerStrings += [p.name]

            if len(players) > 1 and hasattr(self.players[0],'pname') and  hasattr(self.players[1],'pname'):
                for p in players:
                    playerStrings += [p.pname]

            self.vis = vis.visualiser(gridSize=self.players[0].game.gridSize,speed=visSpeed,playerStrings=playerStrings,
                                  resolution=visResolution)

        if trainGames is None:
            nRuns = len(run_games)
        else:
            gens, gens_count, tot_gens = trainGames
            nRuns = gens

        # Play the game a number of times
        for game in range(1, nRuns + 1):
            if trainGames is None:
                if len(players)==1:
                    if game==1:
                        msg = "\nTournament (single-player mode) %s!" % (players[0].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose:
                            sys.stdout.write(msg)
                else:
                    if game==1:
                        msg = "\nTournament %s vs. %s!!!" % (players[0].name, players[1].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose:
                            sys.stdout.write(msg)
                msg = "\n    Game %d..." % (game)
                if self.in_tournament or hasattr(self,'vexcept'):
                    self.game_report.append(msg)

                if self.verbose or hasattr(self,'vexcept'):
                    sys.stdout.write("\n  Game %d..." % (game))

            else:
                msg = "\n  Gen %3d/%d..." % (game+gens_count, tot_gens)
                if self.in_tournament:
                    self.train_report.append(msg)

                if self.verbose:
                    sys.stdout.write(msg)

            if trainGames is None and game in show_games:
                showGame = "Cleaners!"
                if len(self.players) > 1 and hasattr(self.players[0],'pname') and hasattr(self.players[1],'pname'):
                    showGame = '%s vs %s, game %d' % (self.players[0].pname, self.players[1].pname, game)
            else:
                showGame = None

            if trainGames is None and game in save_games:
                saveGame = True
            else:
                saveGame = False

            sgame = CleanersPlay(self,showGame,saveGame)
            gameResult = sgame.play(players)

            if gameResult is None:
                if self.in_tournament:
                    return
                else:
                    print("Error! No game result!")
                    traceback.print_exc()
                    sys.exit(-1)

            if trainGames is None:
                score = gameResult
                if len(players) > 1:
                    if score>0:
                        if hasattr(self.players[0],'pname'):
                            msg = "won by %s with" % (players[0].pname)
                        else:
                            msg = "won by %s (blue) with" % (players[0].name)

                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)
                    elif score<0:
                        if hasattr(self.players[1],'pname'):
                            msg = "won by %s with" % (players[1].pname)
                        else:
                            msg = "won by %s (purlpe) with" % (players[1].name)
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)
                    else:
                        msg = "tied with"
                        if self.in_tournament:
                            self.game_report.append(msg)

                        if self.verbose or hasattr(self,'vexcept'):
                            sys.stdout.write(msg)

                msg = " score=%03d after %d turn" % (np.abs(score),sgame.turn+1)
                if sgame.turn!=0:
                    msg += "s"
                msg += "."

                if self.in_tournament:
                    self.game_report.append(msg)

                if self.verbose or hasattr(self,'vexcept'):
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                self.game_scores[0].append(score)
                if len(players) > 1:
                    self.game_scores[1].append(-score)

            else:
                try:
                    if game + gens_count < tot_gens:
                        players[0].new_generation_agents(game+gens_count)
                    else:
                        players[0].evaluate_fitness()
                except Exception as e:
                    if self.in_tournament:
                        self.game_scores[p].append(-500)
                        self.game_messages[p] = str(e)
                        self.game_play = False
                    else:
                        traceback.print_exc()
                        sys.exit(-1)

        if len(show_games) > 0:
            time.sleep(5)
            del self.vis
            self.vis = None

    # Play visualisation of a saved game
    @staticmethod
    def load(loadGame,visResolution=(720,480), visSpeed='normal'):
        import vis_pygame as vis

        if not os.path.isfile(loadGame):
            print("Error! Saved game file '%s' not found." % loadGame)
            sys.exit(-1)

        # Open the game file and read data
        try:
            with gzip.open(loadGame) as f:
              (player1Name,player2Name,vis_data,gridSize) = pickle.load(f)
        except:
            print("Error! Failed to load %s." % loadGame)

        playerStrings = [player1Name]
        if player2Name is not None:
            playerStrings += [player2Name]

        # Create an instance of visualiser
        v = vis.visualiser(gridSize=gridSize, speed=visSpeed, playerStrings=playerStrings,
                       resolution=visResolution)

        # Show visualisation
        titleStr = "Cleaners! %s" % os.path.basename(loadGame)
        for t,v_data in enumerate(vis_data):
            v.show(v_data, turn=t, titleStr=titleStr)


def main(argv):
    # Load the defaults
    from settings import game_settings

    if not isinstance(game_settings['gridSize'],tuple):
        print("Error! Invalid setting for gridSize.  Must be at tuple (Y,X)")
        sys.exit(-1)


    Y, X = game_settings['gridSize']
    nSquares = Y*X

    if nSquares < 25:
        print("Error! Invalid setting (Y,X)=%s for gridSize.  Y x X must be at least 25" % game_settings['gridSize'])
        sys.exit(-1)

    gridRegions = nSquares//10

    if gridRegions < game_settings['nCleaners']:
        print("Error! Invalid setup with gridSize=%s and nCleaners=%d settings." %  (game_settings['gridSize'],  game_settings['nCleaners']))
        minGridSize = int(np.ceil(np.sqrt(game_settings['nCleaners']*10)))
        maxCleaners = gridRegions
        print("Either increase gridSize to %dx%d total squares or reduce nCleaners to %d." %  (minGridSize,minGridSize,  maxCleaners))
        sys.exit(-1)


    if game_settings['visSpeed'] != 'normal' and game_settings['visSpeed'] != 'fast' and game_settings['visSpeed'] != 'slow':
        print("Error! Invalid setting '%s' for visualisation speed.  Valid choices are 'slow','normal',fast'" % game_settings['visSpeed'])
        sys.exit(-1)

    if not 'player1' in game_settings and not 'player2' in game_settings:
        print("Error! At least one player agent must be specified in settings.py.")
        sys.exit(-1)
    elif not 'player1' in game_settings:
        game_settings['player1'] = game_settings['player2']
        game_settings['player2'] = None
    elif not 'player2' in game_settings:
        game_settings['player2'] = None



    # Create a new game and run it
    g = CleanersGame(gridSize=game_settings['gridSize'],
                nTurns=game_settings['nTurns'],nChargers=game_settings['nCleaners']//2,
                nAgents=game_settings['nCleaners'],
                saveFinalGames=game_settings['saveFinalGames'],
                seed=game_settings['seed'])

    g.run(game_settings['player1'],
          game_settings['player2'],
          visResolution=game_settings['visResolution'],
          visSpeed=game_settings['visSpeed'])

if __name__ == "__main__":

    ##########
    avg_fitnesses = []

    # generation_count = 0
    # fitness_5g = np.zeros((40))

    main(sys.argv[1:])

    #########
    # print(avg_fitnesses)

    # 创建折线图
    plt.plot(range(1,10001), avg_fitnesses)

    # 添加标签和标题
    plt.xlabel('Generation')
    plt.ylabel('Avg_fitness')
    plt.title('Evaluation of average fitness over generations')
    # plt.xticks(range(1,501))

    # 显示图表
    plt.show()

