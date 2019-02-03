To use program  ./run.sh <config_file>

set configure file variables
puzzleFile..................A predefined puzzle file or leave blank for random boards
log..........................The specified log file
solution.....................The specified solution file
seed.........................A user specified seed, 0-no seed specified
wallOnOff....................1-walls requirements honored, 0- wall requirements ignored
forced_validity..............0-no bulbs placed for ever solution, 1- 'must-have' bulbs given to all
defaultRow...................The number of rows for random boards if not specified
defaultCol...................The number of columns for random boards if not specified
maxRows......................The upper bound for rows of a random board
maxCols......................The upper bound for columns of a random board
experiments..................The number of experiments
evaluations..................The number of individual evaluations in each experiment
exit_on_converge.............0-no exit for converged solutions, 1- exit after some number of static generations
number_converge_evals........The number of iterations with same fitness before exit
population_size..............A the number of solution E.A. works with
number_of_children...........The number of children from each generation
tournament_size_parent.......The number of randomly selected parents to compete
tournament_size_survival.....The number of randomly selected parents and children to compete
parent_selection_type........1-rank based truncation, 2-uniform random, 3-fitness proportional, 4-k-tournament w/ replacement
survival_selection_type......1-rank based truncation, 2-uniform random, 3-fitness proportional, 4-k-tournament w/o replacement
penalty_coefficient..........scales the number of invalid instances
fitness_type.................1-basic fitness, 2-penalty fitness, 3-repair fitness
survival_strategy............1-plus, 2-comma
mutation_rate................scales the rate of mutations


There are a lot of command line options if your interested
default(no args) is to run assignment given puzzle data file

-dimSet <%d %d>
gives a random board of %d X %d diminsions

-randBrd 
gives a 10X12 random board

-rectBrd
gives a random rectangle board

-sqrBrd
gives a random square board

-loadConfFile <file string>
load  a specific config files

-loadFile <file string>
allows you to load a specific puzzle

-cliSeed <%d>
allows you to load a specific seed

-brdMode <0 | 1>
turns on and off the wall requirments
