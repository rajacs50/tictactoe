"""
Tic Tac Toe Player
"""

import math
import copy
import numpy as np

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if (board == initial_state()):
        return 'X'
    count = {'X': 0, 'O': 0}
    for row in board:
        if row.count(None) < len(row):
            count['O'] += row.count('O')
            count['X'] += row.count('X')
    return min(count, key=count.get)

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    result = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                result.add((i,j))
    return result


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    temp_board = copy.deepcopy(board)
    curr_player = player(board)
    i, j = action
    if temp_board[i][j] != EMPTY:
        raise Exception('Not a valid move')
    else:
        temp_board[i][j] = curr_player
    return temp_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    np_board = np.array(board)
    pos = ['X', 'O']
    for p in pos:
        # check rows
        if (np.all(np_board == p, axis=1).any()):
            return p
        # check columns
        elif (np.all(np_board == p, axis=0).any()):
            return p
        # check diagonals
        elif (np.all(np.diag(np_board) == p)):
            return p
        # check flipped diagonals
        elif (np.all(np.fliplr(np_board).diagonal() == p)):
            return p

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    e_count = 0
    for row in board:
        if row.count(EMPTY) != 0:
            e_count += 1
    if e_count != 0:
        return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    check_board = winner(board)
    if check_board == 'X':
        return 1
    elif check_board == 'O':
        return -1
    else:
        return 0


def minimax(board, p=None):
    """
    Returns the optimal action for the current player on the board.
    """
    p = p or player(board)
    if (terminal(board)):
        return None
    else:
        possibilities = actions(board)
        outcome = {}
        for places in possibilities:
            next_move = result(board, places)
            if winner(next_move):
                outcome[places] = utility(next_move)
                if p == 'X' and outcome[places] == 1:
                    return places  #, outcome[places]
                elif p == 'O' and outcome[places] == -1:
                    return places  #, outcome[places]
            if terminal(next_move):
                return places  #, 0
            
            # non-leaf; run further minimax and collate the results
            sub_result, outcome[places] = minimax(next_move)

        # choose the optimal action
        max_play = max(outcome, key=outcome.get)
        min_play = min(outcome, key=outcome.get)
        if p == 'X':
            return max_play
        return min_play
