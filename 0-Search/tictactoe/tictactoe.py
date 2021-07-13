"""
Tic Tac Toe Player
"""

import copy
import random
import math

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
    result = X
    plays_count=0
    for row in board:
        for col in row:
            if col:
                plays_count += 1

    if plays_count%2 != 0:
        result = O

    return result


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    result = set()
    board_length = range(len(board))
    for row in board_length:
        for col in board_length:
            if board[row][col] == EMPTY:
                result.add((row, col))

    return result


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    play_move = player(board)
    current_board = copy.deepcopy(board)
    (i, j) = action
    current_board[i][j] = play_move
    return current_board

def has_diagonal_win(board):
    board_length = range(len(board))
    col_count = 0
    last_val = board[0][col_count]
    for row in board_length:
        if board[row][row] == last_val:
            if row == 2:
                return last_val
        else:
            break


    # Here resets the code vars.
    col_count = len(board)-1
    last_val = board[0][col_count]

    for row in board_length:
        if board[row][col_count] == last_val:
            col_count -= 1
            if row == 2:
                return last_val
        else:
            break

    return None

def has_horizontal_win(board):
    board_length = range(len(board))
    win_play = None

    for row in board_length:
        last_val = board[row][0]
        win_play = last_val
        for col in board_length:
            if board[row][col] != last_val:
                win_play = None

        if win_play != None:
            return win_play
    return win_play

def has_vertical_win(board):
    board_length = range(len(board))
    win_play = None

    for row in board_length:
        last_val = board[0][row]
        win_play = last_val
        for col in board_length:
            if board[col][row] != last_val:
                win_play = None

        if win_play != None:
            return win_play
    return win_play


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    return has_vertical_win(board) or has_horizontal_win(board) or has_diagonal_win(board) or None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    result = winner(board)
    if result is None:
        result = True
        board_length = range(len(board))
        for row in board_length:
            for col in board_length:
                if board[row][col] == EMPTY:
                    result = False
                    break

    return result



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    player_turn = player(board)
    if player_turn == X:
        return 1
    elif player_turn == O:
        return -1
    return 0


def max_val(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    val = -math.inf
    best = None
    for action in actions(board):
        m_val = max(val, min_val(result(board, action), alpha, beta)[0])
        alpha = max(alpha, val)

        if m_val > val:
            best = action
            val = m_val

        if alpha >= beta:
            break

    return (val, best)


def min_val(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    val = math.inf
    best = None
    for action in actions(board):
        m_val = min(val, max_val(result(board, action), alpha, beta)[0])
        beta = min(beta, val)

        if m_val < val:
            best = action
            val = m_val

        if beta <= alpha:
            break

    return (val, best)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    This minimax function works with alpha-beta pruning.
    """

    if board == initial_state():
        return (random.randint(0, 2), random.randint(0, 2))
    else:
        current_player = player(board)
        next_move_result = None

        if current_player == X:
            next_move_result = min_val(board, -math.inf, math.inf)[1]
        else:
            next_move_result = max_val(board, -math.inf, math.inf)[1]

    return next_move_result


