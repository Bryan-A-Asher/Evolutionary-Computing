"""
programmer:     Bryan A. Asher
course:         evolutionary computing: comp sci 5401-1a
date:           Friday, September 28, 2018
file:           akari_functions_1c.py
description:    function definitions for akari_1c.py
"""

import copy
import random
import sys
import time
from operator import itemgetter
import numpy as np


# command line and configurations

def cli_handler():
    """description: sets user specified conditions."""

    # all false to start
    rand_board, square_board, rect_board, set_dim, set_seed, board_mode, \
        load_puzzle, load_config = 0, 0, 0, [0], [0], [0], [0], [0, '']

    # load default dims
    if 'randboard' in sys.argv:
        rand_board = 1

    # load random square board
    if 'sqrboard' in sys.argv:
        square_board = 1

    # load random rectangle board
    if 'rectboard' in sys.argv:
        rect_board = 1

    # load specific board dims
    if 'setdim' in sys.argv:
        index = sys.argv.index('setdim')
        set_dim = [1, sys.argv[index + 1], sys.argv[index + 2]]
    else:
        set_dim = [0, 0, 0]

    # load specific seed
    if 'setseed' in sys.argv:
        index = sys.argv.index('setseed')
        set_seed = [1, sys.argv[index + 1]]
    else:
        set_seed = [0, 0]

    # load specific puzzle file
    if 'loadpuzzle' in sys.argv:
        index = sys.argv.index('loadpuzzle')
        load_puzzle = [1, sys.argv[index + 1]]

    # load specific config file
    if 'loadconfig' in sys.argv:
        index = sys.argv.index('loadconfig')
        load_config = [1, sys.argv[index + 1]]
    elif len(sys.argv) > 1:
        if sys.argv[1][-4:] == '.cfg':
            load_config = [1, sys.argv[1]]
    else:
        load_config = [0, 0]

    # turn walls on or off
    if 'boardmode' in sys.argv:
        index = sys.argv.index('boardmode')
        board_mode = [1, sys.argv[index + 1]]
    else:
        board_mode = [0, 0]

    return [rand_board, square_board, rect_board, set_dim, set_seed,
            board_mode, load_puzzle, load_config]


################################################################################


def config_handler(cli_bundle):
    """description: sets the configuration file conditions."""

    board_seed = None
    if cli_bundle[7][0] == 1:
        config_file = 'config_files/' + cli_bundle[7][1]
        config_read = open(config_file, 'r')

    else:
        config_file = 'default.cfg'
        cli_bundle[7][1] = config_file
        config_read = open('config_files/default.cfg', 'r')

    # read in config_read data
    config_values = config_read.read().splitlines()
    config_read.close()

    # set configured variables
    puzzle_file = config_values[0]
    log_file = config_values[1]
    solution_file = config_values[2]
    seed = int(config_values[3])
    walls_on_off = int(config_values[4])
    forced_validity = int(config_values[5])
    default_row = int(config_values[6])
    default_col = int(config_values[7])
    max_rows = int(config_values[8])
    max_cols = int(config_values[9])
    experiments = int(config_values[10])
    evaluations = int(config_values[11])
    exit_on_converge = int(config_values[12])
    number_converge_evals = int(config_values[13])
    population_size = int(config_values[14])
    number_children = int(config_values[15])
    tournament_size_parent = int(config_values[16])
    tournament_size_survival = int(config_values[17])
    parent_selection_type = int(config_values[18])
    survival_selection_type = int(config_values[19])
    penalty_coefficient = float(config_values[20])
    fitness_type = int(config_values[21])
    survival_strategy = int(config_values[22])
    mutation_rate = float(config_values[23])

    # if no puzzle file, set to random
    if puzzle_file == '':
        cli_bundle[0] = 1

    # cli arguments override config specifications
    # turn on/off wall parameter
    if cli_bundle[5][0] == 1:
        walls_on_off = cli_bundle[5][1]
    # seed via cli
    if cli_bundle[4][0] == 1:
        board_seed = int(cli_bundle[4][1])
    # seed via config file
    elif seed != 0:
        board_seed = seed
    # random seed
    elif int(cli_bundle[4][0]) == 0:
        board_seed = int(round(time.perf_counter() * 1000000000))

    return [puzzle_file, log_file, solution_file, board_seed, walls_on_off,
            forced_validity, default_row, default_col, max_rows, max_cols,
            experiments, evaluations, exit_on_converge, number_converge_evals,
            population_size, number_children, tournament_size_parent,
            tournament_size_survival, parent_selection_type, survival_selection_type,
            penalty_coefficient, fitness_type, survival_strategy, mutation_rate, cli_bundle]


# Board Construction and Access


def puzzle_file_to_specs(puzzle_file):
    """Reads puzzle data file into a nested list"""

    # pull data strings from file into list and cut off \n
    if puzzle_file == '':
        return []
    with open(puzzle_file) as read_file:
        board_specs = read_file.read().splitlines()

    # convert the above strings into sublist of integers
    for item in enumerate(board_specs):
        board_specs[item[0]] = list(board_specs[item[0]].split(' '))
        for element in enumerate(board_specs[item[0]]):
            board_specs[item[0]][element[0]] = \
                int(board_specs[item[0]][element[0]])

    return board_specs


################################################################################


def get_empty_space_info(board):
    """ get space coordinates and quantity """

    space_coord = []
    number_spaces = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == ' ':
                space_coord.append([i, j])
                number_spaces += 1

    return space_coord, number_spaces


################################################################################


def get_wall_info(board):
    """get wall coordinates, number and bulb requirements."""

    wall_coord = []
    number_walls = 0
    wall_reqs = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if str(board[i][j]).isdigit():
                if str(board[i][j]) != '5':
                    wall_reqs += int(board[i][j])
                number_walls += 1
                wall_coord.append([i, j, board[i][j]])

    return wall_coord, number_walls, wall_reqs


################################################################################


def get_bulb_info(board):
    """get bulb coordinates and number of bulbs."""

    bulb_coord = []
    number_bulbs = 0
    bulb_reqs = 0

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == chr(42):
                number_bulbs += 1
                bulb_coord.append([i, j])

    return bulb_coord, number_bulbs


################################################################################


def set_bulbs(board, bulb_list):
    """ places bulbs on the board and lights it's path"""

    for ele in enumerate(bulb_list):
        bulb_coord = ele[1]
        row_index = bulb_coord[0]
        col_index = bulb_coord[1]
        board[row_index][col_index] = chr(42)
        board = light_up_path(board, row_index, col_index)

    return board


################################################################################


def sum_wall_bulbs(wall_cord, board, ignore_5=1):
    """checks around walls and counts the bulbs
    wall_cord of the form [row_index, col_index, numBlubs]"""

    bulb_sum = 0

    if ignore_5 == 0:
        ignor_wall_value = 999
    else:
        ignor_wall_value = 5
    # check left
    if wall_cord[1] >= 1 and board[wall_cord[0]][wall_cord[1]] != \
            ignor_wall_value and board[wall_cord[0]][wall_cord[1] - 1] == chr(42):
        bulb_sum += 1
    # check above
    if wall_cord[0] >= 1 and board[wall_cord[0]][wall_cord[1]] != \
            ignor_wall_value and board[wall_cord[0] - 1][wall_cord[1]] == chr(42):
        bulb_sum += 1
    # check right
    if wall_cord[1] <= len(board[wall_cord[0]]) - 2 and \
            board[wall_cord[0]][wall_cord[1]] != ignor_wall_value and \
            board[wall_cord[0]][wall_cord[1] + 1] == chr(42):
        bulb_sum += 1
    # check below
    if wall_cord[0] <= len(board) - 2 and board[wall_cord[0]][wall_cord[1]] != \
            ignor_wall_value and board[wall_cord[0] + 1][wall_cord[1]] == chr(42):
        bulb_sum += 1
    return bulb_sum


################################################################################


def in_light_path(board, cndt_row, cndt_col):
    """detects if a bulb placement is in the path of another bulb"""
    # check left
    for i in range(len(board[cndt_row + 1 - 1][:cndt_col]), -1, -1):
        if board[cndt_row][i] == chr(42):
            return True
        if str(board[cndt_row][i]).isdigit():
            break
    # check right
    for i in range(len(board[cndt_row][cndt_col + 1:])):
        if board[cndt_row][cndt_col + 1 + i] == chr(42):
            return True
        if str(board[cndt_row][cndt_col + 1 + i]).isdigit():
            break
    # check above
    for i in range(len(board[:cndt_row]), -1, -1):
        if board[i][cndt_col] == chr(42):
            return True
        if str(board[i][cndt_col]).isdigit():
            break
    # check below
    for i in range(len(board[cndt_row:])):
        if board[cndt_row + i][cndt_col] == chr(42):
            return True
        if str(board[cndt_row + i][cndt_col]).isdigit():
            break
    return False


################################################################################


def light_up_path(board, row, col):
    """marks the light path of a bulb"""

    # row to the right
    for i in range(len(board[row][col + 1:])):
        if board[row][col + 1 + i] == ' ':
            board[row][col + 1 + i] = chr(34)
        if str(board[row][col + 1 + i]).isdigit():
            break
    # row to the left
    for i in range(len(board[row][:col]), -1, -1):
        if board[row][i] == ' ':
            board[row][i] = chr(34)
        if str(board[row][i]).isdigit():
            break
    # colm path above
    for i in range(len(board[:row]), -1, -1):
        if board[i][col] == ' ':
            board[i][col] = chr(34)
        if str(board[i][col]).isdigit():
            break
    # colm path below
    for i in range(len(board[row:])):
        if board[row + i][col] == ' ':
            board[row + i][col] = chr(34)
        if str(board[row + i][col]).isdigit():
            break
    return board


################################################################################


def clear_light_path(board):
    """ clears lit path markers"""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == chr(34):
                board[i][j] = ' '
    return board


################################################################################


def clear_board(board):
    """ removes all light paths and bulbs from a board"""

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == chr(42) or board[i][j] == chr(34):
                board[i][j] = ' '
    return board


################################################################################


def get_num_lit_sqrs(board):
    """counts lit up squares"""
    lit_up_sqrs = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == chr(34) or board[i][j] == chr(42):
                lit_up_sqrs += 1
    return lit_up_sqrs


################################################################################


def get_num_filled_sqrs(board):
    """count lit up squares"""
    total = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == chr(34) or \
                    board[i][j] == chr(42) or \
                    str(board[i][j]).isdigit():
                total += 1
    return total


################################################################################

def fill_board(board, option):
    """fills a board with chars"""
    for i in range(len(board)):
        for j in range(len(board[0])):
            if option == 'walls':
                board[i][j] = random.randrange(0, 6)
            # add other options later, maybe.
    return board


################################################################################


def build_empty_board(rows, cols):
    """ builds an empty board"""
    board = []
    for i in range(rows):
        board.append([])
        for _ in range(cols):
            board[i].append(' ')
    return board


################################################################################


def build_specified_board(specs):
    """takes board specs and creates a matrix representation"""

    # grab row and col values
    number_cols = specs[0][0]
    number_rows = specs[1][0]

    # convert origin from bottom left to top left
    for i in range(2, len(specs)):
        specs[i][0] -= 1
        specs[i][1] = number_rows - specs[i][1]
        specs[i][0], specs[i][1] = specs[i][1], specs[i][0]

    # create empty board
    board = build_empty_board(number_rows, number_cols)

    # add walls to blank board
    for i in range(2, len(specs)):
        x_coord, y_coord, wall_val = specs[i]
        board[x_coord][y_coord] = wall_val

    return board


################################################################################


def build_solvable_board(board, number_rows, number_cols):
    """build a solvable Akari board"""

    board_specs = []
    bulb_locations = []
    number_empty_spaces = number_rows * number_cols
    number_walls = random.randrange(0, number_empty_spaces)

    for _ in range(number_walls):
        if number_walls == number_empty_spaces:
            return fill_board(board, "walls")

        for _ in range(5):
            # try 5 times to pick random wall placement
            wall_row_indx = random.randrange(0, number_rows)
            wall_col_indx = random.randrange(0, number_cols)

            # check for wall and bulb collisions
            if [wall_col_indx, wall_row_indx, 5] not in board_specs:
                if board[wall_row_indx][wall_col_indx] == ' ':
                    # set specs and board
                    board_specs.append([wall_row_indx, wall_col_indx, 5])
                    board[wall_row_indx][wall_col_indx] = 5
                    number_walls -= 1
                    break

    while get_num_filled_sqrs(board) < number_empty_spaces:

        # place a bulb
        bulb_row_indx = random.randrange(0, number_rows)
        bulb_col_indx = random.randrange(0, number_cols)

        # check for wall and bulb conflict
        if not in_light_path(board, bulb_row_indx, bulb_col_indx):
            if board[bulb_row_indx][bulb_col_indx] == ' ':
                board[bulb_row_indx][bulb_col_indx] = chr(42)
                board = light_up_path(board, bulb_row_indx, bulb_col_indx)
                bulb_locations.append((bulb_row_indx, bulb_col_indx))

    # count bulbs around walls and set specs and board appropriatly
    for wall in board_specs:
        bulbs_around_wall = sum_wall_bulbs(wall, board, 0)
        wall[2] = bulbs_around_wall
        board[wall[0]][wall[1]] = wall[2]

    # remove bulbs
    for bulb in bulb_locations:
        board[bulb[0]][bulb[1]] = ' '

    # clear light up path
    board = clear_light_path(board)

    return board


################################################################################


def build_random_board(max_rows, max_cols, cli_brd_type, seed_param):
    """This function builds a random solvable board
       with several user definable dimension parameters"""

    random.seed(seed_param)

    if cli_brd_type[0] == 'rectboard':
        # get random dimensions
        num_rows = random.randrange(1, max_rows + 1)
        num_cols = random.randrange(1, max_cols + 1)
        board = build_empty_board(num_rows, num_cols)
        board = build_solvable_board(board, num_rows, num_cols)
    elif cli_brd_type[0] == 'sqrboard':
        # get random dimensions
        row_eqs_col = random.randrange(1, max_cols + 1)
        num_rows = row_eqs_col
        num_cols = row_eqs_col
        board = build_empty_board(num_rows, num_cols)
        board = build_solvable_board(board, num_rows, num_cols)
    elif cli_brd_type[0] == 'randboard':
        # get random dimensions
        num_cols = int(cli_brd_type[1])
        num_rows = int(cli_brd_type[2])
        board = build_empty_board(num_rows, num_cols)
        board = build_solvable_board(board, num_rows, num_cols)
    elif cli_brd_type[0] == 'setdim':
        # get random dimensions
        num_rows = int(cli_brd_type[1][0])
        num_cols = int(cli_brd_type[1][1])
        board = build_empty_board(num_rows, num_cols)
        board = build_solvable_board(board, num_rows, num_cols)

    return board


################################################################################


def build_specs_from_board(board, item):
    """
    takes a board and outputs the specifications for that board
    in non-offset reversed format. If item = 1, get wall specs;
    If item = 2 get bulb specs.
    """

    if board == []:
        return []

    # get coordinates of info
    wall_info = get_wall_info(board)
    wall_coord = wall_info[0]

    bulb_info = get_bulb_info(board)
    bulb_coord = bulb_info[0]

    if item == 1:
        obj = wall_coord
    elif item == 2:
        obj = bulb_coord

    # get dimensions of board
    number_rows = len(board)
    number_cols = len(board[0])

    # convert origin from top left to bottom left

    for i in range(len(obj)):
        # convert col coordinate to col number
        obj[i][1] += 1

        # convert row coordinate to row number
        obj[i][0] = abs(number_rows - obj[i][0])

        # switch row and col position
        obj[i][0], obj[i][1] = obj[i][1], obj[i][0]

    # construct final info
    if item == 1:
        specs = [[number_cols], [number_rows]] + obj
    elif item == 2:
        specs = obj

    return specs


################################################################################


def board_handler():
    """
    given a configuration, the function selects the appropriate type
    of board to build.
    """

    # get configuration information
    config_bundle = config_handler(cli_handler())

    puzzle_file = config_bundle[0]
    seed = config_bundle[3]
    default_row = config_bundle[6]
    default_col = config_bundle[7]
    max_rows = config_bundle[8]
    max_cols = config_bundle[9]
    cli_bundle = config_bundle[-1]
    rand_board = cli_bundle[0]
    square_board = cli_bundle[1]
    rect_board = cli_bundle[2]
    set_dim = cli_bundle[3]
    load_puzzle = cli_bundle[6]

    # put Akari board specs from file into list
    board_data = puzzle_file_to_specs(puzzle_file)

    # create Akari board
    if puzzle_file != '':
        board = build_specified_board(board_data)
    # load cli specified file
    if load_puzzle[0] == 1:
        board_data = puzzle_file_to_specs(load_puzzle[1])
        board = build_specified_board(board_data)
    # build random specific dimension board
    elif set_dim[0] == 1:
        cli_param = ('setdim', set_dim[1:])
        board = build_random_board(max_rows, max_cols, cli_param, seed)
    # build random rectangle board
    elif rect_board == 1:
        board = build_random_board(max_rows, max_cols, ('rectboard', 1), seed)
    # build random square board
    elif square_board == 1:
        board = build_random_board(max_rows, max_cols, ('sqrboard', 1), seed)
    # build random board with dims in config_read file default 10 x 12
    elif rand_board == 1:
        cli_param = ('randboard', default_col, default_row)
        board = build_random_board(max_rows, max_cols, cli_param, seed)
    else:
        board = None

    return board


################################################################################


def force_bulbs_on_board(board, walls_coord):
    """ loop through walls and check if open spaces match wall requirement """
    eval_board = copy.deepcopy(board)
    forced_list = []
    f_bulbs = 1
    while f_bulbs > 0:

        f_bulbs -= 1
        for i in range(len(walls_coord)):
            space_sum = 0
            left, above, right, below = 0, 0, 0, 0
            # check left
            if walls_coord[i][1] >= 1 and \
                    eval_board[walls_coord[i][0]][walls_coord[i][1] - 1] == ' ':
                space_sum += 1
                left = 1

            # check above
            if walls_coord[i][0] >= 1 and \
                    eval_board[walls_coord[i][0] - 1][walls_coord[i][1]] == ' ':
                space_sum += 1
                above = 1

            # check right
            if walls_coord[i][1] <= len(eval_board[walls_coord[i][0]]) - 2 and \
                    eval_board[walls_coord[i][0]][walls_coord[i][1] + 1] == ' ':
                space_sum += 1
                right = 1

            # check below
            if walls_coord[i][0] <= len(eval_board) - 2 and \
                    eval_board[walls_coord[i][0] + 1][walls_coord[i][1]] == ' ':
                space_sum += 1
                below = 1

            # walls that get forced bulbs

            if space_sum == eval_board[walls_coord[i][0]][walls_coord[i][1]]:
                if left == 1 and not wall_conflict(eval_board, walls_coord[i][0], walls_coord[i][1] - 1):
                    eval_board[walls_coord[i][0]][walls_coord[i][1] - 1] = chr(42)
                    forced_list.append((walls_coord[i][0], walls_coord[i][1] - 1))
                    f_bulbs = 1

                if above == 1 and not wall_conflict(eval_board, walls_coord[i][0] - 1, walls_coord[i][1]):
                    eval_board[walls_coord[i][0] - 1][walls_coord[i][1]] = chr(42)
                    forced_list.append((walls_coord[i][0] - 1, walls_coord[i][1]))
                    f_bulbs = 1

                if right == 1 and not wall_conflict(eval_board, walls_coord[i][0], walls_coord[i][1] + 1):
                    eval_board[walls_coord[i][0]][walls_coord[i][1] + 1] = chr(42)
                    forced_list.append((walls_coord[i][0], walls_coord[i][1] + 1))
                    f_bulbs = 1

                if below == 1 and not wall_conflict(eval_board, walls_coord[i][0] + 1, walls_coord[i][1]):
                    eval_board[walls_coord[i][0] + 1][walls_coord[i][1]] = chr(42)
                    forced_list.append((walls_coord[i][0] + 1, walls_coord[i][1]))
                    f_bulbs = 1
    return eval_board, forced_list


################################################################################


def set_bulbs_randomly(num_bulbs, space_coords):
    """generates random bulb coordinates"""

    individual = set()
    # randomly pick bulb placements
    for _ in range(num_bulbs):
        pick = random.choice(space_coords)
        individual.add((pick[0], pick[1]))

    individual = list(individual)
    return individual


################################################################################


def wall_conflict(board, row_index, col_index):
    """returns True if the addition of a bulb invalidates a solution"""

    num_good_placements = 0
    adj_wall_index = []

    # check left
    if col_index >= 1 and str(board[row_index][col_index - 1]).isdigit():
        adj_wall_index.append([row_index, col_index - 1, board[row_index][col_index - 1]])

    # check above
    if row_index >= 1 and str(board[row_index - 1][col_index]).isdigit():
        adj_wall_index.append([row_index - 1, col_index, board[row_index - 1][col_index]])

    # check right
    if col_index <= len(board[row_index]) - 2 and \
            str(board[row_index][col_index + 1]).isdigit():
        adj_wall_index.append([row_index, col_index + 1, board[row_index][col_index + 1]])

    # check below
    if row_index <= len(board) - 2 and \
            str(board[row_index + 1][col_index]).isdigit():
        adj_wall_index.append([row_index + 1, col_index, board[row_index + 1][col_index]])

    for i in range(len(adj_wall_index)):
        if sum_wall_bulbs(adj_wall_index[i], board) < int(adj_wall_index[i][2]):
            num_good_placements += 1

    if num_good_placements == len(adj_wall_index):
        return False

    return True


################################################################################


def wall_collision(board, row_index, col_index):
    """checks for bulbs being placed on top of walls"""

    if str(board[row_index][col_index]).isdigit():
        return True
    return False


################################################################################
################### Constraint Evolutionary Algorithm ##########################
################################################################################


def init_population(num_of_individuals, board, forced_validity):
    """ produces a set of randomly generated individuals """

    population = []
    forced_list = []
    # auxiliary information
    wall_info = get_wall_info(board)
    walls_coord = wall_info[0]

    if forced_validity == 1:
        board, forced_list = force_bulbs_on_board(board, walls_coord)

    space_info = get_empty_space_info(board)

    spaces_coord = space_info[0]
    number_spaces = space_info[1]

    # get a list of random bulb counts that each individual will draw form.
    bulbs = []
    while len(bulbs) < num_of_individuals:
        number_bulbs = random.randrange(1, number_spaces + 1)
        bulbs.append(number_bulbs)

    # build individuals
    for i in range(len(bulbs)):
        individual = set_bulbs_randomly(bulbs[i], spaces_coord)
        population.append(individual + forced_list)

    return population


################################################################################


def validate_individual(board, individual, wall_on_off=1):
    """
    checks for invalid individuals and returns true if valid, false otherwise.
    """

    for i in range(len(individual)):
        row_index = individual[i][0]
        col_index = individual[i][1]

        if wall_collision(board, row_index, col_index):  # wall Collision
            return False, board
        if board[row_index][col_index] == chr(42):  # bulb Collision
            return False, board
        if in_light_path(board, row_index, col_index):  # bulb Conflict
            return False, board
        if int(wall_on_off) == 1 and \
                wall_conflict(board, row_index, col_index):  # wall Conflict
            return False, board
        board = set_bulbs(board, [(individual[i][0], individual[i][1])])

    return True, board


################################################################################


def basic_fitness(board, individual, wall_on_off):
    """ gives a numeric score proportional to an individuals solution quality"""

    # keep board clean
    eval_board = copy.deepcopy(board)

    # get wall info
    wall_info = get_wall_info(eval_board)
    wall_coord = wall_info[0]

    # get board info
    space_info = get_empty_space_info(board)
    number_spaces = space_info[1]

    # check for collisions, conflicts
    valid, eval_board = validate_individual(eval_board, individual, wall_on_off)

    # invalid individuals have a fitness of zero
    if not valid:
        return [individual, 0.0], [individual, 0.0, eval_board]

    total_wall_bulbs = 0
    for i in range(len(wall_coord)):
        total_wall_bulbs += sum_wall_bulbs(wall_coord[i], eval_board)

    # calculate fitness score
    lit_up_spaces = get_num_lit_sqrs(eval_board)
    fitness = lit_up_spaces / number_spaces

    return [individual, fitness], [individual, fitness, eval_board]


################################################################################


def fitness_with_penalty(board, individual, wall_on_off, penalty_coefficient):
    """
    gives a numeric score proportional to an individuals solution quality,
    while considering invalid individuals
    """

    eval_board = copy.deepcopy(board)

    # get wall info
    wall_info = get_wall_info(eval_board)
    wall_coord = wall_info[0]

    # get board info
    space_info = get_empty_space_info(board)
    number_spaces = space_info[1]

    # check for number of collisions, conflicts and set board with bulbs
    # and lit paths

    num_invalid_instances, eval_board = \
        count_invalid_instances(eval_board, individual, wall_on_off)

    # check wall adjacency accuracy
    total_wall_bulbs = 0
    for i in range(len(wall_coord)):
        total_wall_bulbs += sum_wall_bulbs(wall_coord[i], eval_board)

    # calculate fitness score
    lit_up_spaces = get_num_lit_sqrs(eval_board)
    penalty = num_invalid_instances * penalty_coefficient

    fitness = (lit_up_spaces - penalty) / number_spaces
    # no negative fitnesses
    if fitness < 0:
        fitness = 0.0

    return [individual, fitness], [individual, fitness, eval_board]


################################################################################


def fitness_with_repair(board, individual):
    """
    gives a numeric score proportional to an individuals solution quality,
    while repairing invalid individuals
    """

    eval_board = copy.deepcopy(board)

    # get wall info
    wall_info = get_wall_info(eval_board)
    wall_coord = wall_info[0]

    # get board info
    space_info = get_empty_space_info(board)
    number_spaces = space_info[1]

    # repair individual, if need be.
    individual = repair(eval_board, individual)

    # every individual has at least one bulb
    if individual == []:
        placement = random.randrange(len(space_info))
        individual.append((space_info[0][placement][0], space_info[0][placement][1]))

        # set board
    eval_board = copy.deepcopy(board)
    eval_board = set_bulbs(eval_board, individual)

    # check wall adjacency accuracy
    total_wall_bulbs = 0
    for i in range(len(wall_coord)):
        total_wall_bulbs += sum_wall_bulbs(wall_coord[i], eval_board)

    # calculate fitness score
    lit_up_spaces = get_num_lit_sqrs(eval_board)
    fitness = lit_up_spaces / number_spaces
    return [individual, fitness], [individual, fitness, eval_board]


################################################################################


def fitness_evaluation(board, individual, penalty_coefficient, fitness_type=1, \
                       wall_on_off=1):
    """selects the appropriate fitness function given a fitness type"""

    results = []
    if fitness_type == 1:
        results = basic_fitness(board, individual, wall_on_off)

    elif fitness_type == 2:
        results = fitness_with_penalty(board, individual, wall_on_off, \
                                       penalty_coefficient)

    elif fitness_type == 3:
        results = fitness_with_repair(board, individual)

    return results


################################################################################


def count_invalid_instances(board, individual, wall_on_off=1, true_count=0):
    """
    count the number of times an individual violates a constraint and
    returns a count and set board
    """

    collision_flag = 0
    eval_board = copy.deepcopy(board)

    if true_count == 0:
        increment = 10000  # used to kill the fitness score of
        # individuals who have collisions

    elif true_count == 1:  # used to return actual number of invalid cases
        increment = 1

    count = 0
    for i in range(len(individual)):
        row_index = individual[i][0]
        col_index = individual[i][1]
        if wall_collision(eval_board, row_index, col_index):  # wall Collision
            count += increment
            collision_flag = 1

        elif eval_board[row_index][col_index] == chr(42):  # bulb Collision
            count += increment
            collision_flag = 1

        if collision_flag == 0:
            eval_board = set_bulbs(eval_board, [(individual[i][0], \
                                                 individual[i][1])])

        if in_light_path(eval_board, row_index, col_index):  # bulb Conflict
            count += 1

        elif int(wall_on_off) == 1 and \
                wall_conflict(eval_board, row_index, col_index):  # wall Conflict
            count += 1

    return count, eval_board


################################################################################


def move_bulb(board, individual, row_index, col_index):
    """tries to move invalid bulb placements to valid placements"""

    working_board = copy.deepcopy(board)

    problematic_bulb_index = individual.index((row_index, col_index))
    working_board = set_bulbs(working_board, individual[:problematic_bulb_index])

    # can we move up?
    if row_index - 1 >= 0:
        valid = validate_individual(working_board, [(row_index - 1, col_index)])
        if valid[0]:
            return (row_index - 1, col_index)
    # can we move down?
    if row_index + 1 < len(working_board):
        valid = validate_individual(working_board, [(row_index + 1, col_index)])
        if valid[0]:
            return (row_index + 1, col_index)
    # can we move left?
    if col_index - 1 >= 0:
        valid = validate_individual(working_board, [(row_index, col_index - 1)])
        if valid[0]:
            return (row_index, col_index - 1)
    # can we move right?
    if col_index + 1 < len(working_board[0]):
        valid = validate_individual(working_board, [(row_index, col_index + 1)])
        if valid[0]:
            return (row_index, col_index + 1)
    # no, delete it
    return (-1, -1)


################################################################################


def repair(board, individual):
    """
    checks for invalid individuals and repairs if found.
    """
    count, set_board = count_invalid_instances(board, individual, 1, 1)
    eval_board = copy.deepcopy(board)

    while count > 0:

        removed = 0

        valid = validate_individual(board, individual)
        if valid[0]:
            break

        if removed >= len(individual):
            break

        untested_position = len(individual)
        for i in range(untested_position):

            row_index = individual[i - removed][0]
            col_index = individual[i - removed][1]

            if wall_collision(board, row_index, col_index):  # wall Collision
                new_placement = move_bulb(board, individual, row_index, col_index)
                if new_placement == (-1, -1):
                    individual.remove((row_index, col_index))
                    count -= 1
                    removed += 1
                else:
                    individual[i - removed] = new_placement
                    eval_board = set_bulbs(eval_board, [new_placement])
                    count -= 1


            elif board[row_index][col_index] == chr(42):  # bulb Collision
                new_placement = move_bulb(board, individual, row_index, col_index)
                if new_placement == (-1, -1):
                    individual.remove((row_index, col_index))
                    count -= 1
                    removed += 1
                else:
                    individual[i - removed] = new_placement
                    eval_board = set_bulbs(eval_board, [new_placement])
                    count -= 1


            elif in_light_path(board, row_index, col_index):  # bulb Conflict
                new_placement = move_bulb(board, individual, row_index, col_index)
                if new_placement == (-1, -1):
                    individual.remove((row_index, col_index))
                    count -= 1
                    removed += 1
                else:
                    individual[i - removed] = new_placement
                    eval_board = set_bulbs(eval_board, [new_placement])
                    count -= 1


            elif wall_conflict(board, row_index, col_index):  # wall Conflict
                new_placement = move_bulb(board, individual, row_index, col_index)
                if new_placement == (-1, -1):
                    individual.remove((row_index, col_index))
                    count -= 1
                    removed += 1
                else:
                    individual[i - removed] = new_placement
                    eval_board = set_bulbs(eval_board, [new_placement])
                    count -= 1

    return individual


################################################################################


def parent_selection(population, tournament_size_parent, parent_selection_type):
    """takes a population and selects parents based on parent_section_type"""

    parents = []
    parents_with_fitness = []
    number_of_parents = len(population) / 2

    # order by fitness(rank them)
    population.sort(key=lambda x: x[1], reverse=True)

    if parent_selection_type == 1:
        for i in range(int(len(population) / 2)):
            parents.append(population[i][0])
            parents_with_fitness.append(population[0])

    elif parent_selection_type == 2:
        # uniform random
        for i in range(int(number_of_parents)):
            pick = random.randint(0, len(population) - 1)
            parents.append(population[pick][0])
            parents_with_fitness.append(population[pick])


    elif parent_selection_type == 3:
        # fitness proportional selection

        # get all fitness values in a list, must add to unity
        fitness_vals = []
        for i in range(len(population)):
            fitness_vals.append(population[i][1])

        # get total fitness
        total_fitness = 0
        for j in range(len(population)):
            total_fitness += fitness_vals[j]

        if total_fitness == 0:
            total_fitness = 0.000000000001
        # get proportional fitness
        for k in range(len(population)):
            population[k][1] = population[k][1] / total_fitness
            fitness_vals[k] /= total_fitness

        # sample parents with chance of picking each parent relative
        # to its proportional fitness
        parents_index = np.random.choice(len(population), \
                                         int(number_of_parents), p=fitness_vals)

        for l in range(len(parents_index)):
            parents.append(population[parents_index[l]][0])
            parents_with_fitness.append(population[parents_index[l]])

    elif parent_selection_type == 4:
        # tournament
        while len(parents) < number_of_parents:
            contenders = []
            for i in range(tournament_size_parent):
                pick = random.randint(0, len(population) - 1)
                contenders.append(population[pick])
            contenders.sort(key=lambda x: x[1], reverse=True)
            parents.append(contenders[0][0])
            parents_with_fitness.append(contenders[0])

    return parents, parents_with_fitness


################################################################################


def replacement(population_size, parents, offspring, tournament_size_survival, \
                survival_selection_type, survival_strategy):
    """
    takes a parents and offspring populations and selects next generation based
    on survial_section_type and survival_stratigy
    """

    population = []
    pool = []
    population_with_fitness = []
    if survival_strategy == 1:  # plus
        pool = parents + offspring
    elif survival_strategy == 2:  # comma
        pool = offspring

    if survival_selection_type == 1:  # truncation

        pool.sort(key=lambda x: x[1], reverse=True)
        for i in range(population_size):
            population.append(pool[i][0])
            population_with_fitness.append(pool[i])
        return population, population_with_fitness

    if survival_selection_type == 2:  # uniform random

        random.shuffle(pool)
        for i in range(population_size):
            pick = random.randint(0, population_size - 1)
            population.append(pool[pick][0])
            population_with_fitness.append(pool[i])
        return population, population_with_fitness

    if survival_selection_type == 3:  # fitness proportional

        # get total fitness
        total_fitness = 0

        for ele in enumerate(pool):
            total_fitness += ele[1][1]

        temp = []
        if total_fitness == 0:
            total_fitness = 0.0000000000001
        for ele in pool:
            temp1 = ele[0]
            temp2 = ele[1] / total_fitness
            temp.append([temp1, temp2])
        pool = temp

        # get all fitness values in a list, must add to unity
        fitness_values = []
        for i in range(len(pool)):
            fitness_values.append(pool[i][1])

        # sample pool with chance of picking each member relative
        # to its proportional fitness

        population_index = np.random.choice(len(pool), \
                                            population_size, p=fitness_values)

        for i in range(len(population_index)):
            population.append(pool[population_index[i]][0])
            population_with_fitness.append(pool[population_index[i]])

        return population, population_with_fitness

    if survival_selection_type == 4:  # k-tournament w/o replacement
        # tournament
        while len(population) < population_size:
            contenders = []

            for i in range(tournament_size_survival):
                if population_size == len(population):
                    continue
                pick = random.randint(0, len(pool) - 1)
                contenders.append(pool[pick])

            contenders.sort(key=lambda x: x[1], reverse=True)

            if len(contenders) > 0:
                winner_index = pool.index(contenders[0])
                pool.pop(winner_index)
                population.append(contenders[0][0])
                population_with_fitness.append(contenders[0])

        return population, population_with_fitness

    return population, population_with_fitness


################################################################################


def variation(parents, number_of_children, board):
    """ a recombination operator, parent + parent = child """

    offspring = []

    for i in range(number_of_children):

        # select partners
        parent_one = parents[random.randrange(len(parents))]
        parent_two = parents[random.randrange(len(parents))]

        # select crossover points
        horiz_crossover = random.randint(1, len(board))

        # build offspring
        # we need to partition the individual
        parent1_partitions = [[], []]
        parent2_partitions = [[], []]

        # partition parents according to the above crossover row
        for j in range(len(parent_one)):
            if parent_one[j][0] < horiz_crossover:
                parent1_partitions[0].append(parent_one[j])
            else:
                parent1_partitions[1].append(parent_one[j])

        for k in range(len(parent_two)):
            if parent_two[k][0] < horiz_crossover:
                parent2_partitions[0].append(parent_two[k])
            else:
                parent2_partitions[1].append(parent_two[k])

        child = []
        for l in range(2):
            if random.randint(0, 2) == 0:
                child = child + parent1_partitions[l]
            else:
                child = child + parent2_partitions[l]

        # don't allow empty children, randomly select parent to clone
        if child == []:
            if random.randint(0, 2) == 0:
                child = copy.deepcopy(parent_one)
            else:
                child = copy.deepcopy(parent_two)

        offspring.append(child)

    return offspring


################################################################################


def mutation(offspring, board, mutation_rate):
    """randomly changes a bulb's position with a 1/(number of bulbs) chance."""

    # move a bulb somewhere else, maybe.

    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if random.random() < mutation_rate:
                new_row = random.randrange(len(board))
                new_col = random.randrange(len(board[0]))
                if not (new_row, new_col) in offspring[i]:
                    offspring[i][j] = (new_row, new_col)

    for i in range(len(offspring)):
        if offspring[i] == []:
            break
        if random.random() < mutation_rate / 2:
            pick = random.randrange(len(offspring[i]))
            offspring[i].pop(pick)
    return offspring


################################################################################


def write_log_headers(config_bundle):
    """creates and writes headers to log files """

    with open(config_bundle[1], 'a+') as file_out:

        # randomly generated puzzle
        if config_bundle[0] == '':
            file_out.write('Randomly Generated Puzzle\n')

            # puzzle file
        elif config_bundle[0] != '':
            file_out.write('configuration File: ' + str(config_bundle[24][7][1]) + '\n')
            file_out.write('Puzzle File: ' + str(config_bundle[0][11:]) + '\n')

        file_out.write('log = ' + str(config_bundle[1]) + '\n')
        file_out.write('solution = ' + str(config_bundle[2]) + '\n')
        file_out.write('seed = ' + str(config_bundle[3]) + '\n')
        file_out.write('wallOnOff = ' + str(config_bundle[4]) + '\n')
        file_out.write('forced_validity = ' + str(config_bundle[5]) + '\n')
        file_out.write('defaultRow = ' + str(config_bundle[6]) + '\n')
        file_out.write('defaultCol = ' + str(config_bundle[7]) + '\n')
        file_out.write('maxRows = ' + str(config_bundle[8]) + '\n')
        file_out.write('maxCols = ' + str(config_bundle[9]) + '\n')
        file_out.write('experiments = ' + str(config_bundle[10]) + '\n')
        file_out.write('evaluations = ' + str(config_bundle[11]) + '\n')
        file_out.write('exit_on_converge = ' + str(config_bundle[12]) + '\n')
        file_out.write('number_converge_evals = ' + str(config_bundle[13]) + '\n')
        file_out.write('population_size = ' + str(config_bundle[14]) + '\n')
        file_out.write('number_of_children = ' + str(config_bundle[15]) + '\n')
        file_out.write('tournament_size_parent = ' + str(config_bundle[16]) + '\n')
        file_out.write('tournament_size_survival = ' + str(config_bundle[17]) + '\n')
        file_out.write('parent_selection_type = ' + str(config_bundle[18]) + '\n')
        file_out.write('survival_selection_type = ' + str(config_bundle[19]) + '\n')
        file_out.write('penalty_coefficient = ' + str(config_bundle[20]) + '\n')
        file_out.write('fitness_type = ' + str(config_bundle[21]) + '\n')
        file_out.write('survival_strategy = ' + str(config_bundle[22]) + '\n')
        file_out.write('mutation_rate = ' + str(config_bundle[23]) + '\n\n')
        file_out.write('Result Log\n\n')


################################################################################


def write_eval_log(num_evaluations, population_with_fitness, config_bundle, \
                   experiment_number, eval_flag):
    """writes the new max fitness score and average population fitness to log"""

    average_fitness = get_average_fitness(population_with_fitness)
    population_with_fitness.sort(key=lambda x: x[1], reverse=True)

    if eval_flag == 0:
        with open(config_bundle[1], 'a+') as file_out:
            file_out.write('\nRun ' + str(experiment_number) + '\n')

    if eval_flag != 0:
        with open(config_bundle[1], 'a+') as file_out:
            file_out.write(str(num_evaluations) + '\t' + str(average_fitness) + \
                           '\t' + str(population_with_fitness[0][1]) + '\n')

    file_out.close()


################################################################################


def write_solution_log(ifb_colection, solution_log):
    """writes max solution over all runs to solution log """

    max_fitness = 0
    max_index = 0
    for i in range(len(ifb_colection)):
        if float(ifb_colection[i][1]) > max_fitness:
            max_fitness = ifb_colection[i][1]
            max_index = i
    board = ifb_colection[max_index][2]
    number_lit_up = get_num_lit_sqrs(board)

    wall_specs = build_specs_from_board(board, 1)
    bulb_specs = build_specs_from_board(board, 2)

    dims = wall_specs[:2]
    wall_specs = sorted(wall_specs[2:], key=itemgetter(0, 1))
    wall_specs = dims + wall_specs
    bulb_specs = sorted(bulb_specs, key=itemgetter(0, 1))

    with open(solution_log, 'w') as file_out:
        # print board
        #     for i in range(len(board)):
        #         for j in range(len(board[i])):
        #             file_out.write(str(board[i][j]) + ' ')
        #         file_out.write('\n')
        #     file_out.write('\n')

        # write puzzle board
        for i in range(len(wall_specs)):
            for j in range(len(wall_specs[i])):
                file_out.write(str(wall_specs[i][j]) + ' ')
            file_out.write('\n')

        # write number of lit spaces
        file_out.write(str(number_lit_up) + '\n')

        # write bulb coordinates
        for i in range(len(bulb_specs)):
            for j in range(len(bulb_specs[i])):
                file_out.write(str(bulb_specs[i][j]) + ' ')
            file_out.write('\n')


################################################################################


def get_average_fitness(population):
    """ calculates the average fitness of the population"""

    total_fitness = 0
    for i in range(len(population)):
        total_fitness += population[i][1]

    return total_fitness / len(population)


################################################################################


def constraint_evo():
    """ a constraint optimization algorithm for Akari."""

    # general set up
    cli_info = cli_handler()
    config_info = config_handler(cli_info)

    # supporting information
    solution_log = config_info[2]
    seed = config_info[3]
    walls_on_off = config_info[4]
    forced_validity = config_info[5]
    experiments = config_info[10]
    max_evaluations = config_info[11]
    exit_on_converge = config_info[12]
    number_converge_evals = config_info[13]
    population_size = config_info[14]
    number_of_children = config_info[15]
    tournament_size_parent = config_info[16]
    tournament_size_survival = config_info[17]
    parent_selection_type = config_info[18]
    survival_selection_type = config_info[19]
    penalty_coefficient = config_info[20]
    fitness_type = config_info[21]
    survival_strategy = config_info[22]
    mutation_rate = config_info[23]

    random.seed(seed)
    # for  log tracking, persists across all experiments
    # [individual, max_score]
    global_max_score = [0, 0]
    header_flag = 0

    for exp in range(experiments):

        # initialization
        board = board_handler()
        print(np.matrix(board), 'con-evo')
        population = init_population(population_size, board, forced_validity)

        # for termination and log tracking, persists across generations
        # [max_sore, gen_@_max_score]
        local_max_score = [0, 0]
        old_gen = 0

        # number of initial evaluations is mu
        num_evaluations = population_size

        # start cycle
        number_generation = 1
        eval_flag = 0

        while num_evaluations < max_evaluations:

            # evaluation
            pop_ifb_collection = []
            population_with_fitness = []
            for i in range(population_size):
                fitness, pop_indvid_fit_board = fitness_evaluation(board, \
                                                                   population[i], penalty_coefficient, fitness_type,
                                                                   walls_on_off)
                population_with_fitness.append(fitness)
                pop_ifb_collection.append(pop_indvid_fit_board)

            # get logs up and running
            if eval_flag == 0:
                if header_flag == 0:
                    write_log_headers(config_info)
                write_eval_log(num_evaluations, population_with_fitness, \
                               config_info, exp + 1, eval_flag)

                write_solution_log(pop_ifb_collection, solution_log)
                header_flag = 1
                eval_flag = 1

            number_generation += 1

            # parent selection
            parents, parents_with_fitness = \
                parent_selection(population_with_fitness, tournament_size_parent, \
                                 parent_selection_type)

            # variation
            offspring = variation(parents, number_of_children, board)

            # mutation
            offspring = mutation(offspring, board, mutation_rate)

            # evaluation
            offspring_with_fitness = []
            off_ifb_collection = []
            for j in range(len(offspring)):
                fitness, off_indvid_fit_board = fitness_evaluation(board, \
                                                                   offspring[j], penalty_coefficient, fitness_type,
                                                                   walls_on_off)
                offspring_with_fitness.append(fitness)
                off_ifb_collection.append(off_indvid_fit_board)

            # survivor selection
            population, population_with_fitness = replacement(population_size, \
                                                              parents_with_fitness, offspring_with_fitness, \
                                                              tournament_size_survival, survival_selection_type,
                                                              survival_strategy)

            # writ logs and track evaluations

            if header_flag != 0:
                write_eval_log(num_evaluations, population_with_fitness, \
                               config_info, exp + 1, eval_flag)
                eval_flag = 1

            num_evaluations += (len(offspring))

            # display
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            average_fitness = get_average_fitness(population_with_fitness)

            print('Evaluation:', num_evaluations, \
                  '  Average Fitness:', average_fitness, \
                  'Best Fitness:', population_with_fitness[0][1])

            total_collection = pop_ifb_collection + off_ifb_collection

            # update global max, maybe.
            if global_max_score[1] < population_with_fitness[0][1]:
                global_max_score[0] = population_with_fitness[0][0]
                global_max_score[1] = population_with_fitness[0][1]

                write_solution_log(total_collection, solution_log)

            # terminate on convergence
            if population_with_fitness[0][1] == 1.0:
                local_max_score[0] = population_with_fitness[0][1]
                local_max_score[1] = number_generation
                break

            #  exit on convergence val and mutate on premature convergence
            if population_with_fitness[0][1] > local_max_score[0]:
                local_max_score[0] = population_with_fitness[0][1]
                local_max_score[1] = number_generation
                old_gen = number_generation

            elif population_with_fitness[0][1] == local_max_score[0]:
                if number_generation - old_gen >= number_converge_evals and \
                        exit_on_converge == 1:
                    break

################################################################################
