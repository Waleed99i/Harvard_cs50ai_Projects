import math
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    x_count = 0
    o_count = 0
    for row in board:
        for cell in row:
            if cell == X:
                x_count += 1
            elif cell == O:
                o_count += 1
    if x_count <= o_count:
        return X
    else:
        return O

def actions(board):
    available_moves = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                available_moves.add((i, j))
    return available_moves

def result(board, action):
    if board[action[0]][action[1]] is not EMPTY:
        raise Exception("Invalid move")
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def checkRows(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    return False

def checkColumns(board, player):
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    return False

def checkTopToBottomDiagonal(board, player):
    if all(board[i][i] == player for i in range(3)):
        return True
    return False

def checkBottomToTopDiagonal(board, player):
    if all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def isTie(board):
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True

def winner(board):
    if checkRows(board, X) or checkColumns(board, X) or checkTopToBottomDiagonal(board, X) or checkBottomToTopDiagonal(board, X):
        return X
    elif checkRows(board, O) or checkColumns(board, O) or checkTopToBottomDiagonal(board, O) or checkBottomToTopDiagonal(board, O):
        return O
    else:
        return None

def terminal(board):
    if winner(board) is not None or isTie(board):
        return True
    return False

def utility(board):
    winning_player = winner(board)
    if winning_player == X:
        return 1
    elif winning_player == O:
        return -1
    else:
        return 0

def minimax(board):
    if terminal(board):
        return None

    current = player(board)

    if current == X:
        value = -math.inf
        best_move = None
        for action in actions(board):
            move_val = min_value(result(board, action))
            if move_val > value:
                value = move_val
                best_move = action
        return best_move
    else:
        value = math.inf
        best_move = None
        for action in actions(board):
            move_val = max_value(result(board, action))
            if move_val < value:
                value = move_val
                best_move = action
        return best_move

def max_value(board):
    if terminal(board):
        return utility(board)
    value = -math.inf
    for action in actions(board):
        value = max(value, min_value(result(board, action)))
    return value

def min_value(board):
    if terminal(board):
        return utility(board)
    value = math.inf
    for action in actions(board):
        value = min(value, max_value(result(board, action)))
    return value
