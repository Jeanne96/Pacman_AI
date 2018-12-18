# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    # print (legalMoves)
    #print bestScore
    # print bestIndices
    # print chosenIndex'''
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
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    #print successorGameState
    #print newPos
    #print newFood
    #print newGhostStates
    print newScaredTimes

    #find the minimum distance  to ghosts
    MinDistToGhost = 20
    for ghostState in newGhostStates:
        distGhost = manhattanDistance(newPos,ghostState.getPosition())
        if distGhost < MinDistToGhost:
	    MinDistToGhost = distGhost

    #find the minumum distance to foods
    FoodState = currentGameState.getFood()
    MinDistToFood = 20
    for food in FoodState.asList():
        distFood = manhattanDistance(newPos, food)
        if distFood < MinDistToFood:
	    MinDistToFood = distFood
    
    evaluation = -( MinDistToFood*(0.4) +  MinDistToGhost*(-0.4) + len(newFood.asList())* (0.2))
   # print evaluation
    return evaluation

   # print distGhost   
   # print successorGameState.getWalls().height
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
                
    #max-value(state)
    def max_value(state, depth):
        depth = depth + 1
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = -float('Inf')
        for action in state.getLegalActions(0):
            v = max(v, min_value(state.generateSuccessor(0, action), depth, 1))
        return v
    #min-value(state)
    def min_value(state, depth, ghostNum):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float('Inf')
        for action in state.getLegalActions(ghostNum):
            if ghostNum == gameState.getNumAgents() - 1:
                v = min(v, max_value(state.generateSuccessor(ghostNum, action), depth))
            else:
                v = min(v, min_value(state.generateSuccessor(ghostNum, action), depth, ghostNum + 1))
        return v
      
    
    legalMoves = gameState.getLegalActions(0)
    maximum = -float('Inf')
    pacmanAction = ''
    newActions = []
    for action in legalMoves:
        if action != Directions.STOP:
            newActions.append(action)
    for action in newActions:
        depth = 0
        currentMax = min_value(gameState.generateSuccessor(0, action), depth, 1)
        if currentMax > maximum:
            maximum = currentMax
            pacmanAction = action
    #print pacmanAction
    return pacmanAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    def max_value(state, alpha, beta, depth):
        depth = depth + 1
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = -float('Inf')
        for action in state.getLegalActions(0):
            v = max(v, min_value(state.generateSuccessor(0, action), alpha, beta, depth, 1))
            
            alpha = max(alpha, v)
            if alpha >= beta:
                return v
        return v

    def min_value(state, alpha, beta, depth, ghostNum):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float('Inf')
        for action in state.getLegalActions(ghostNum):
            if ghostNum == gameState.getNumAgents() - 1:
                v = min(v, max_value(state.generateSuccessor(ghostNum, action), alpha, beta, depth))
            else:
                v = min(v, min_value(state.generateSuccessor(ghostNum, action), alpha, beta, depth, ghostNum + 1))
            
            beta = min(beta, v)
            if alpha >= beta:
                return v
        return v
      

    legalMoves = gameState.getLegalActions(0)
    maximum = -float('Inf')
    alpha = -float('Inf')
    beta = float('Inf')
    pacmanAction = ''
    newActions = []
    for action in legalMoves:
    
        depth = 0
        currentMax = min_value(gameState.generateSuccessor(0, action), alpha, beta, depth, 1)
        if currentMax > maximum:
            maximum = currentMax
            pacmanAction = action
        return pacmanAction
  
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
    def max_value(state, depth):
        depth = depth + 1
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        v = -float('Inf')
        for action in state.getLegalActions(0):
            v = max(v, exp_value(state.generateSuccessor(0, action), depth, 1))
        return v

    def exp_value(state, depth, ghostNum):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = 0
        p =  len(state.getLegalActions(ghostNum))
        
        for action in state.getLegalActions(ghostNum):
            if ghostNum == gameState.getNumAgents() - 1:
                v = v + (max_value(state.generateSuccessor(ghostNum, action), depth)) / p 
            else:
                v = v + (exp_value(state.generateSuccessor(ghostNum, action), depth, ghostNum + 1)) / p 
        return v
       
    
    legalMoves = gameState.getLegalActions(0)
    maximum = -float('Inf')
    pacmanAction = ''
    newActions = []
    for action in legalMoves:
        if action != Directions.STOP:
            newActions.append(action)
    for action in newActions:
        depth = 0
        currentMax = exp_value(gameState.generateSuccessor(0, action), depth, 1)
        if currentMax > maximum or (currentMax == maximum and random.random() > .3):
            maximum = currentMax
            pacmanAction = action
    return pacmanAction

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

