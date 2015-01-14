# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        score = 0
        minfoodDist = 999999
        for food in newFood.asList():
            minfoodDist = min(manhattanDistance(food, newPos), minfoodDist)

        #check number of food Pellets pacman has to eat
        foodPellets = currentGameState.getCapsules()
        if newPos in foodPellets:
            score += 1000000

        for ghostState in newGhostStates:
                ghostDist = manhattanDistance(newPos, ghostState.getPosition())
                #if ghost is closer to Pacman and scaredTimer is 0
                #reduce the score and ghost will eat Pacman
                if ghostDist < 3.0 and ghostState.scaredTimer == 0:
                        score -= ghostDist * 1000
                #if ghost dist to Pacman is 0 and scaredTimer is > 0
                #Pacman is invincible.
                #increase the score as its best case for Pacman
                elif ghostDist == 0 and ghostState.scaredTimer > 0:
                        score += 1000000
                #if ghost dist to Pacman is < than scaredTimer
                #and state.scaredTimer is expiring soon then still Pacman is invincible.
                #increase the score by small amount as Pacman has still time to breath
                elif ghostState.scaredTimer > 0 and ghostDist < ghostState.scaredTimer:
                        score += (1.0 / (1.0 + ghostDist))


        score += 10.0 / (1.0 + len(newFood.asList()))  + (1.0 /(1000 + minfoodDist))

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        v = float('-inf')
        bestAction = Directions.STOP
        #start minimax with Pacman
        for action in gameState.getLegalActions(PACMAN_INDEX):
            val = self.minValue(1, 0, gameState.generateSuccessor(PACMAN_INDEX, action))
            if val > v and action != Directions.STOP:
                v = val
                bestAction = action

        return bestAction


    def maxValue(self, agent, depth,  state):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('-inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                    val = self.minValue(agent+1, depth, state.generateSuccessor(agent, action))
                    if val > v:
                        v = val
            return v

    def minValue(self, agent, depth, state):
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                if agent == state.getNumAgents() - 1:
                    #next turn is Pacmans
                    val = self.maxValue(PACMAN_INDEX, depth+1, state.generateSuccessor(agent, action))
                    if val < v:
                        v = val
                else:
                    #next turn is ghosts
                    val = self.minValue(agent+1, depth, state.generateSuccessor(agent, action))
                    if val < v:
                        v = val
            return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        v = float('-inf')
        alpha= float('-inf')
        beta= float('inf')
        bestAction = Directions.STOP

        actions = gameState.getLegalActions(PACMAN_INDEX)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        #start minimax with Pacman
        for action in actions:
            val = self.minValue(1, 0, gameState.generateSuccessor(PACMAN_INDEX, action),alpha, beta)
            if val > v:
                v = val
                bestAction = action
            if v > beta:
                return bestAction
            alpha = max(v, alpha)

        return bestAction

    def maxValue(self, agent, depth, state, alpha, beta):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)

            if len(actions) > 0:
                v = float('-inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                v = max(v, self.minValue(agent+1, depth, state.generateSuccessor(agent, action), alpha, beta))
                #minimizer will not allow value greater the beta
                if v > beta:
                    return v
                alpha = max(v, alpha)
            return v

    def minValue(self, agent, depth, state, alpha, beta):
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            if len(actions) > 0:
                v = float('inf')
            else:
                v = self.evaluationFunction(state)

            for action in actions:
                if agent == state.getNumAgents() - 1:
                    #next turn is Pacman
                    v = min(v, self.maxValue(PACMAN_INDEX, depth+1, state.generateSuccessor(agent, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(v, beta)
                else:
                    #next turn is ghosts
                    v = min(v, self.minValue(agent+1, depth, state.generateSuccessor(agent, action), alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(v, beta)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        v = float('-inf')
        bestAction = Directions.STOP

        actions = gameState.getLegalActions(PACMAN_INDEX)
        for action in actions:
                val = self.expValue(1, 0, gameState.generateSuccessor(PACMAN_INDEX, action))
                if val > v and action != Directions.STOP:
                    v = val
                    bestAction = action

        return bestAction


    def maxValue(self, agent, depth, state):
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            actions = state.getLegalActions(agent)
            if len(actions) > 0:
                v = float('-inf')
            else:
                v = self.evaluationFunction(state)

            for action in state.getLegalActions(agent):
                v = max(v, self.expValue(agent+1, depth, state.generateSuccessor(agent, action)))

            return v

    def expValue(self, agent, depth, state):
        PACMAN_INDEX = 0  #agent index 0 means Pacman
        if depth == self.depth:
            return self.evaluationFunction(state)
        else:
            v = 0;
            actions = state.getLegalActions(agent)

            for action in actions:
                if agent == state.getNumAgents() - 1:
                    #its pacman turn
                    v += self.maxValue(PACMAN_INDEX, depth+1, state.generateSuccessor(agent, action))
                else:
                    #its ghost turn
                    v += self.expValue(agent+1, depth, state.generateSuccessor(agent, action))

            if len(actions) != 0:
                return v / len(actions)
            else:
                return self.evaluationFunction(state)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    minfoodDist = 999999
    for food in newFood.asList():
        minfoodDist = min(manhattanDistance(food, newPos), minfoodDist)

    score = 0
    #check number of food Pellets pacman has to eat
    foodPellets = currentGameState.getCapsules()

    for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            #if ghost is closer to Pacman and scaredTimer is 0
            #reduce the score and ghost will eat Pacman
            if ghostDist < 3.0 and ghostState.scaredTimer == 0:
                score -= ghostDist * 1000
            #if ghost dist to Pacman is 0 and scaredTimer is > 0
            #Pacman is invincible.
            #increase the score as its best case for Pacman
            elif ghostDist == 0 and ghostState.scaredTimer > 0:
                score += 100000
            #if ghost dist to Pacman is < than scaredTimer
            #and state.scaredTimer is expiring soon then still Pacman is invincible.
            #increase the score by small amount as Pacman has still time to breath
            elif ghostState.scaredTimer > 0 and ghostDist < ghostState.scaredTimer:
                score += (1.0 / (1.0 + ghostDist))


    score += 10.0 / (1.0 + len(newFood.asList())) + (1.0 /(1000 + minfoodDist)) \
             + 10.0 / (1.0 + len(foodPellets))  + currentGameState.getScore()

    return score;


# Abbreviation
better = betterEvaluationFunction

























