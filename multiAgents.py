"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Student name:
Student ID:

"""

# multiAgents.py
# --------------
# Attribution Information: part of the code were created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# http://ai.berkeley.edu.
# We thank them for that! :)


import random, util, math

from connect4 import Agent


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 1  # agent is always index 1
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class BestRandom(MultiAgentSearchAgent):

    def getAction(self, gameState):
        return gameState.pick_best_move()


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 1)
    """

    def getMinMaxScore(self, gameState, depth):
        if gameState.is_terminal() or depth == 0:
            return None, self.evaluationFunction(gameState)
        turn = gameState.turn
        legal_actions = gameState.getLegalActions()
        # if the turn is agent
        if turn == 1:
            score_max = -math.inf
            action_max = None
            for action in legal_actions:
                state = gameState.generateSuccessor(turn, action)
                # switch turn
                state.switch_turn(turn)
                action1, current_score = self.getMinMaxScore(state, depth - 1)
                if current_score > score_max:
                    action_max = action
                    score_max = current_score
            return action_max, score_max
        else:
            score_min = math.inf
            action_min = None
            for action in legal_actions:
                state = gameState.generateSuccessor(turn, action)
                # switch turn
                state.switch_turn(turn)
                action1, current_score = self.getMinMaxScore(state, depth - 1)
                if current_score < score_min:
                    action_min = action
                    score_min = current_score
            return action_min, score_min

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.isWin():
        Returns whether or not the game state is a winning state for the current turn player

        gameState.isLose():
        Returns whether the game state is a losing state for the current turn player

        gameState.is_terminal()
        Return whether that state is terminal
        """
        action, score = self.getMinMaxScore(gameState, self.depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        turn = gameState.turn
        a, v = self.max_value(gameState, self.depth, -math.inf, math.inf)
        return a

    def max_value(self, state, depth, alpha, beta):
        if state.is_terminal() or depth == 0:
            return None, self.evaluationFunction(state)
        else:
            turn = state.turn
            max_v = -math.inf
            legal_actions = state.getLegalActions()
            max_action = None
            for action in legal_actions:
                s = state.generateSuccessor(turn, action)
                # switch turn
                s.switch_turn(turn)
                current_score = max(max_v, self.min_value(s, depth - 1, alpha, beta)[1])
                if current_score > beta:
                    return action, current_score
                elif current_score > max_v:
                    max_v = current_score
                    max_action = action
                alpha = max(alpha, current_score)
        return max_action, max_v

    def min_value(self, state, depth, alpha, beta):
        if state.is_terminal() or depth == 0:
            return None, self.evaluationFunction(state)
        else:
            turn = state.turn
            min_v = math.inf
            legal_actions = state.getLegalActions()
            min_action = None
            for action in legal_actions:
                s = state.generateSuccessor(turn, action)
                # switch turn
                s.switch_turn(turn)
                current_score = min(min_v, self.max_value(s, depth - 1, alpha, beta)[1])
                if current_score < alpha:
                    return action, current_score
                elif current_score < min_v:
                    min_v = current_score
                    min_action = action
                beta = min(beta, current_score)
        return min_action, min_v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 3)
    """

    def max_val(self, state, depth):
        turn = state.turn
        max_v = -math.inf
        max_a = None
        actions = state.getLegalActions()
        for action in actions:
            s = state.generateSuccessor(turn, action)
            s.switch_turn(turn)
            a, v = self.value(s, depth - 1)
            if max_v < v:
                max_v = v
                max_a = action
        return max_a, max_v

    def exp_val(self, state, depth):
        turn = state.turn
        val = 0
        actions = state.getLegalActions()
        act_ind = random.randint(0, len(actions) - 1)
        act = actions[act_ind]
        for action in actions:
            p = 1 / len(actions)
            s = state.generateSuccessor(turn, action)
            s.switch_turn(turn)
            a, v = self.value(s, depth - 1)
            val += (p * v)
        return act, val

    def value(self, state, depth):
        if state.is_terminal() or depth == 0:
            return None, self.evaluationFunction(state)
        turn = state.turn
        if turn == 1:  # max
            # state.switch_turn(turn)
            return self.max_val(state, depth)
        else:
            # state.switch_turn(turn)
            return self.exp_val(state, depth)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        a, v = self.value(gameState, self.depth)
        return a
