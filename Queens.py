import random
import math
from multiprocessing import Process
import time


# /////////////////////////////////////////////////////////////////////////////////////////////////
# Board Class Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
class Board:
    """Class to store the board and cost of the board.

    Stores the board in a N size integer List.
    Where the index of the integer is what column,
    and the integer value is the row index.
    It stores the cost on initialisation
    so each board only calculates cost once.

    Attributes:
        board (int)[]: Stores the board in List representation.
        cost (int):    Stores the Fitness/Cost of the board.

    Printing:
        How the Board class is used.

        >>> X = Board([x for x in range(8)])
        >>> print(X)
        Q # # # # # # #
        # Q # # # # # #
        # # Q # # # # #
        # # # Q # # # #
        # # # # Q # # #
        # # # # # Q # #
        # # # # # # Q #
        # # # # # # # Q
        Cost = 28
        Board = [0, 1, 2, 3, 4, 5, 6, 7]

    """

    def __init__(self, board):
        self.board = board
        self.cost = cost(self.board)

    def __str__(self):
        output = ""
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if i == self.board[j]:
                    output += "Q "  # Queen is at i,j
                else:
                    output += "- "
            output += "\n"
        output += "Cost = " + str(self.cost) + "\n"
        output += "Board = [" + ", ".join([str(x) for x in self.board]) + "]"
        return output  # Multiline string to display board, with Cost and List structure.

    def __lt__(self, other):
        return self.cost < other.cost  # Used to sort a List of boards
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Board Class End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# cost Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def cost(board):
    """Calculate Fitness/Cost of given board.

    Finds all clashes with each queen left to right.
    Using reletive positioning to find if any queens are on diagonals,
    and checks if they lay on the same horizontal.

    Args:
        board (int)[]: A board represented; index as column and value as row position.

    Returns:
        (int): A count of all the queens that are clashing with another.

    Examples:
        An Example showing for a board size N = 8 getting the cost.

        >>> print(cost([x for x in range(8)]))
        28
    """
    cost = 0  # Stores the total cost to return.
    for i in range(len(board)-1):  # For all queens ignoring last since there is no queen after it
        # Reletive positions.
        diff_up = board[i] - i
        diff_down = board[i] + i
        for j in range(i+1, len(board)):  # Compares all other queens and checks their reletive pos.
            # These three checks, check if any queen to the right
            # lie on the same diagonal +/-, or on the same row.
            if diff_up == board[j] - j:
                cost += 1
            elif diff_down == board[j] + j:
                cost += 1
            if board[i] == board[j]:
                cost += 1
    return cost
# /////////////////////////////////////////////////////////////////////////////////////////////////
# cost Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# get_neighbours Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def get_neighbours(X, N):
    """Will return a List of tuples of all possible coordinates for queens to move to.

    Loops through all board positions and adds coordinates tuples if a
    queen is not in that location.

    Args:
        X (Board): A Board object to get all possible next moves.
        N (int): Size for the N-Queens problem.

    Returns:
        ((int),(int))[]: List of tuples of valid queen moves.

    Examples:
        An Example showing the list of tuples for a given Board.

        >>> X = Board([x for x in range(8)])
        >>> print(get_neighbours(X, 8))
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
         (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
         (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
         (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7),
         (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7),
         (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7),
         (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7),
         (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)]
    """
    neighbours = []
    for i in range(N):
        for j in range(N):
            if (i, j) != (i, X.board[i]):
                neighbours.append((i, j))
    return neighbours
# /////////////////////////////////////////////////////////////////////////////////////////////////
# get_neighbours Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# get_best_neighbours Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def get_best_neighbours(X, N):
    """Will return a Board List of all the best possible moves for a given board.

    Loops through all possible moves and if they have a smaller cost
    then the current min cost they are set into a empty List of Boards.
    Else if they are the same cost they are added to the List of Boards.
    Once all moves are checked it returns the List of Boards.

    Args:
        X (Board): A Board object to get all the best neighbours.
        N (int): Size for the N-Queens problem.

    Returns:
        (Board)[]: List of Boards with valid, but lowest cost neighbours.

    Examples:
        An Example of getting the best neighbours and picking a random one from it.
        The Initial state X is just queens along the diagonal.

        >>> X = Board([x for x in range(8)])
        >>> neighbours = get_best_neighbours(X, 8)
        >>> neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
        >>> print(neighbour)
        Q - - - - - - -
        - - - - - - - -
        - - Q - - - - -
        - - - Q - - - -
        - Q - - Q - - -
        - - - - - Q - -
        - - - - - - Q -
        - - - - - - - Q
        Cost = 22
        Board = [0, 4, 2, 3, 4, 5, 6, 7]
    """
    neighbours = []
    min_cost = X.cost - 1  # Sets the min cost to less than current board's (X's) cost.
    for i in range(N):
        for j in range(N):
            # Checks if new position doesn't have a queen.
            # Then compares costs to build best neighbours List.
            if (i, j) != (i, X.board[i]):
                new_board = list(X.board)
                new_board[i] = j
                next_X = Board(new_board)  # next_X represents contender for next move.
                # If next_X cost is < min_cost the list is reset before adding,
                # else if the cost is the same as current min_cost add it to the List.
                if next_X.cost < min_cost:
                    neighbours = []
                    neighbours.append(next_X)
                    min_cost = next_X.cost
                elif next_X.cost == min_cost:
                    neighbours.append(next_X)
    return neighbours
# /////////////////////////////////////////////////////////////////////////////////////////////////
# get_best_neighbours Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# hill_climb Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def hill_climb(N):
    """Hill climb algorithm (Steepest ascent), to solve N-Queens problem.

    Generate random initial state X then;
    Until solved or there is no more better neighbours,
    Get best neighbour (randomly pick one if there is multiple)
    set current state to neighbour chosen. Then generate new best
    neighbours. Repeat.

    Args:
        N (int): Size for the N-Queens problem.

    Examples:
        Once it finishes it will print in this format.

        >>> hill_climb(4)
        ____________________________Random Restart Hill Climb__________________________________
        Time:         0.0004999637603759766  Seconds.
                Initial State:
        - - - -
        - - - Q
        Q - - -
        - Q Q -
        Cost = 3
        Board = [2, 3, 3, 1]
                End State:
        - Q - -
        - - - Q
        Q - - -
        - - Q -
        Cost = 0
        Board = [2, 0, 3, 1]
        _______________________________________________________________________________________
    """
    # Generates a initial_state.
    X = Board([random.randint(0, N-1) for x in range(N)])
    neighbours = get_best_neighbours(X, N)
    initial_state = X
    # Set Initial time for run time calculations.
    start_time = time.time()
    while X.cost > 0 and len(neighbours) > 0:
        # Gets a random best neighbour if there is any better neighbours.
        neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
        X = neighbour
        neighbours = get_best_neighbours(X, N)  # Gets new set of better neighbours.
    # Prints the algorithm results.
    print("____________________________________Hill Climb_________________________________________")
    print("Time:        ", time.time() - start_time, " Seconds.")
    print("\tInitial State:")
    print(initial_state)
    print("\tEnd State:   ")
    print(X)
    print("_______________________________________________________________________________________")
# /////////////////////////////////////////////////////////////////////////////////////////////////
# hill_climb Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# random_restart_hill_climb Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def random_restart_hill_climb(N):
    """Random restart Hill climb algorithm (Steepest ascent), to solve N-Queens problem.

    Generate random initial state X then;
    Until there is a solution, do the same as hill_climb,
    but if the length of the best neighbours is empty.
    Restart with a random state.

    Args:
        N (int): Size for the N-Queens problem.

    Examples:
        Once it finishes it will print in this format.

        >>> random_restart_hill_climb(4)
        ____________________________Random Restart Hill Climb__________________________________
        Time:         0.0  Seconds.
        Restarts:      1
                Initial State:
        Q - Q -
        - - - -
        - Q - Q
        - - - -
        Cost = 2
        Board = [0, 2, 0, 2]
                End State:
        - - Q -
        Q - - -
        - - - Q
        - Q - -
        Cost = 0
        Board = [1, 3, 0, 2]
        _______________________________________________________________________________________
    """
    # Generates a initial_state.
    X = Board([random.randint(0, N-1) for x in range(N)])
    neighbours = get_best_neighbours(X, N)
    initial_state = X
    # Set Initial time for run time calculations and restarts for messuring performance.
    restarts = 0
    restart_max = 10000
    start_time = time.time()
    while X.cost > 0 and restarts < restart_max:
        while len(neighbours) > 0:
            # Gets a random best neighbour if there is any better neighbours.
            neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
            X = neighbour
            neighbours = get_best_neighbours(X, N)  # Gets new set of better neighbours.
        # Once there is no more neighbours (So no more better moves),
        # if the cost is not equal to goal state "0" randomly set state and try again.
        if X.cost > 0:
            X = Board([random.randint(0, N-1) for x in range(N)])
            neighbours = get_best_neighbours(X, N)
        restarts += 1
    # Prints the algorithm results.
    print("____________________________Random Restart Hill Climb__________________________________")
    print("Time:        ", time.time() - start_time, " Seconds.")
    print("Restarts:    ", restarts)
    print("\tInitial State:")
    print(initial_state)
    print("\tEnd State:   ")
    print(X)
    print("_______________________________________________________________________________________")
# /////////////////////////////////////////////////////////////////////////////////////////////////
# random_restart_hill_climb Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# simulated_annealing Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def simulated_annealing(T, alpha, K, N):
    """Simulated Annealing algorithm, solve N-Queens by cooling T (Temperature).

    Generates a random initial_state, then gets a List of tuples
    for valid moves from the get_neighbours function.
    This List of valid moves is then randomly picked from and a new state is created.
    If the new states cost is better or equal to the current state,
    set the new state to current state. Otherwise accept with probability e^(-(E'-E)/T)
    Where:
        e = Euler number. (2.71828...)
        E = cost of current state
        E' = cost of new state
        T = Temperature. Cooled after K new state checks (by alpha).
    This is repeated K times then T is set to T*alpha. If the goal cost is reached,
    the possible neighbours is empty, or T is smaller than the threshold (T_min).

    Args:
        T (float): Initial temperature to be cooled.
        alpha (float): Cooling rate for T.
        K (int): Loops before T is cooled.
        N (int): Size for the N-Queens problem.

    Examples:
        Once it finishes it will print in this format.

        >>> simulated_annealing(1.0, 0.99, 10, 4)
        _______________________________Simulated Annealing_____________________________________
        Time:         0.0010006427764892578  Seconds.
        Temperature:  0.9509900498999999
                Initial State:
        - - - -
        - Q Q -
        - - - Q
        Q - - -
        Cost = 3
        Board = [3, 1, 1, 2]
                End State:
        - Q - -
        - - - Q
        Q - - -
        - - Q -
        Cost = 0
        Board = [2, 0, 3, 1]
        _______________________________________________________________________________________
    """
    T_min = 0.000001  # Smallest T can get in order to stop high calculation errors.
    X = Board([random.randint(0, N-1) for x in range(N)])
    # Generates a initial_state.
    neighbours = get_neighbours(X, N)
    initial_state = X
    # Set Initial time for run time calculations.
    start_time = time.time()
    while X.cost > 0 and T > T_min and len(neighbours) > 0:
        for k in range(K):
            # Creates a new state from neighbours randomly and gets the cost.
            neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
            new_board = list(X.board)
            new_board[neighbour[0]] = neighbour[1]
            next_X = Board(new_board)
            # If the next_X (new state) is <= X (current state) set X = next_X & get neighbours.
            if next_X.cost <= X.cost:
                X = next_X
                neighbours = get_neighbours(X, N)
            # Calculates probability of accepting higher cost state. Then accepts or skips.
            elif math.e**(-1*(next_X.cost-X.cost)/T) > random.uniform(0, 1):
                X = next_X
                neighbours = get_neighbours(X, N)
            # Break out of the k loop if there is no possible moves left or found goal state.
            if len(neighbours) == 0 or X.cost == 0:
                break
        # Lowers the T (Temperature) by multiplying by alpha (the cooling rate).
        T = T*alpha
    # Prints the algorithm results.
    print("_______________________________Simulated Annealing_____________________________________")
    print("Time:        ", time.time() - start_time, " Seconds.")
    print("Temperature: ", T)
    print("\tInitial State:")
    print(initial_state)
    print("\tEnd State:   ")
    print(X)
    print("_______________________________________________________________________________________")
# /////////////////////////////////////////////////////////////////////////////////////////////////
# simulated_annealing Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# generate_initial_population Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def generate_initial_population(pool_size, N):
    """Generates a population (size = pool_size) of randomly set Board objects of N size.

    Loops pool_size times creating random N sized Board objects.
    Then returns a list containing all the Boards.

    Args:
        pool_size (int): Size of the population to generate.
        N (int): Size for the N-Queens problem.

    Returns:
        (Board)[]: List of randomly set Board objects.

    Examples:
        Showing the cost of all the generated board to show they are random
        with 10 size and N=4.

        >>>for X in generate_initial_population(10, 4):
        >>>    print(X.cost, end=", ")
        3, 4, 2, 2, 3, 5, 3, 3, 1, 4,
    """
    boards = []
    for i in range(pool_size):
        x = Board([random.randint(0, N-1) for x in range(N)])
        boards.append(x)
    return boards
# /////////////////////////////////////////////////////////////////////////////////////////////////
# generate_initial_population Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# genetic Function Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
def genetic(pool_size, selection_factor, mutate_chance, N):
    """Genetic algorithm, to solve N-Queens using parents and children.

    Generates a random initial population of pool_size, then sorts by cost.
    Until there is one board of goal cost "0" or generations is too large (max_generations);
    Select two parents, skewing towards the lower cost parents with pool_size*x^selection_factor.
    This forces better parents to have higher chance of being selected for breeding.
        A random_splice point is taken in order to cut the parents in ~half,
    then two children are made by concatenating opposite halves of the parents.
        After the children boards have been made, they have a chance of being
    mutated (mutate_chance). This just randomly moves a queen up or down (sometimes keeps).
    These two new children are added to the new_population until there is pool_size
    children. This new_population is then set as the population (and sorted) then it repeats.

    Args:
        pool_size (int): Size of the population.
        selection_factor (int): Factor to skew. The > it is the more chance for better parents.
        mutate_chance (float): The probability of a child mutating.
        N (int): Size for the N-Queens problem.

    Examples:
        Once it finishes it will print in this format.

        >>>genetic(20, 5, 0.8, 4)
        ____________________________________Genetic____________________________________________
        Time:         0.0
        Generations:  2
                Initial State:
        - - - Q
        - Q - -
        Q - - -
        - - Q -
        Cost = 1
        Board = [2, 1, 3, 0]
                End State:
        - Q - -
        - - - Q
        Q - - -
        - - Q -
        Cost = 0
        Board = [2, 0, 3, 1]
        _______________________________________________________________________________________

    """
    # Generates initial population and sets initial_state to best initial Board.
    max_generations = 10000  # This is to stop too many generations.
    population = generate_initial_population(pool_size, N)
    generations = 1
    population.sort()
    initial_state = population[0]  # population[0] is the best child of that generation.
    # Set Initial time for run time calculations.
    start_time = time.time()
    while population[0].cost > 0 and generations < max_generations:
        new_population = []
        for i in range(int(pool_size/2)):
            # Picks two randomly (skewed to the better parents) selected parents.
            parent_one = population[int(pool_size*random.random()**selection_factor)]
            parent_two = population[int(pool_size*random.random()**selection_factor)]
            # Gets a random_splice point. Needs to cut the parents so 1-N-2 (inclusivly).
            random_splice = random.randint(1, N-2)
            # Splices the children with opposite halves of the parents.
            child_one_board = parent_one.board[0:random_splice] + parent_two.board[random_splice:N]
            child_two_board = parent_two.board[0:random_splice] + parent_one.board[random_splice:N]
            # Mutates the children with mutate_chance for each.
            if random.random() < mutate_chance:
                child_one_board[random.randint(0, N-1)] = random.randint(0, N-1)
            if random.random() < mutate_chance:
                child_two_board[random.randint(0, N-1)] = random.randint(0, N-1)
            # Creates the children Board objects then appends them to the new_population.
            child_one = Board(child_one_board)
            child_two = Board(child_two_board)
            new_population.append(child_one)
            new_population.append(child_two)
        # The population new_population then sorted.
        population = new_population
        population.sort()
        generations += 1
    # Prints the algorithm results.
    print("____________________________________Genetic____________________________________________")
    print("Time:        ", time.time() - start_time)
    print("Generations: ", generations)
    print("\tInitial State:")
    print(initial_state)
    print("\tEnd State:   ")
    print(population[0])
    print("_______________________________________________________________________________________")
# /////////////////////////////////////////////////////////////////////////////////////////////////
# genetic Function End
# /////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////
# Main Start
# /////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    # This function is just a demonstration on how to use the algorithms.
    # A small UI was created and it is self explanatory. So it will not be documented.
    print("_______________________________________________________________________________________")
    print("What would you like to do?")
    print("1. Hill Climbing")
    print("2. Random Restart")
    print("3. Simulated Annealing")
    print("4. Genetic Algorithm")
    print("5. All in parallel")
    print("Q. Quit")
    print("_______________________________________________________________________________________")
    type = input()
    N = 4
    if type == "1":
        print("Size of the problem?")
        N = int(input())
        if N < 4 or N > 100:
            print("Invalid N range please use 4-100. Quitting...")
            exit()
        hill_climb(N)
    elif type == "2":
        print("Size of the problem?")
        N = int(input())
        if N < 4 or N > 100:
            print("Invalid N range please use 4-100. Quitting...")
            exit()
        random_restart_hill_climb(N)
    elif type == "3":
        print("Size of the problem?")
        N = int(input())
        if N < 4 or N > 100:
            print("Invalid N range please use 4-100. Quitting...")
            exit()
        print("Would you like to enter you own parameters? (Y|N)")
        custom = input()
        if custom == "Y" or custom == "y":
            print("Intital Temperature:")
            T = float(input())
            print("Alpha:")
            alpha = float(input())
            print("K interations:")
            K = int(input())
            simulated_annealing(T, alpha, K, N)
        elif custom == "N" or custom == "n":
            simulated_annealing(1.0, 0.99, 100, N)
        else:
            print("Quiting...")
            exit()
    elif type == "4":
        print("Size of the problem?")
        N = int(input())
        if N < 4 or N > 100:
            print("Invalid N range please use 4-100. Quitting...")
            exit()
        print("Would you like to enter you own parameters? (Y|N)")
        custom = input()
        if custom == "Y" or custom == "y":
            print("Population size:")
            pool_size = int(input())
            print("Selection Factor:")
            selection_factor = int(input())
            print("Mutation Chance:")
            mutate_chance = float(input())
            genetic(pool_size, selection_factor, mutate_chance, N)
        elif custom == "N" or custom == "n":
            genetic(50, 5, 0.8, N)
        else:
            print("Quiting...")
            exit()
    elif type == "5":
        print("Size of the problem?")
        N = int(input())
        if N < 4 or N > 100:
            print("Invalid N range please use 4-100. Quitting...")
            exit()
        # Using multiprocessing library to run in parallel.
        ga = Process(target=genetic, args=(50, 5, 0.8, N))
        ga.start()
        sa = Process(target=simulated_annealing, args=(1.0, 0.99, 100, N))
        sa.start()
        hc = Process(target=hill_climb, args=(N,))
        hc.start()
        rs = Process(target=random_restart_hill_climb, args=(N,))
        rs.start()
        ga.join()
        sa.join()
        hc.join()
        rs.join()
    elif type == "Q" or type == "q":
        print("Quiting...")
        exit()
    else:
        print("Sorry, invalid choice, Quitting...")
        exit()
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Main End
# /////////////////////////////////////////////////////////////////////////////////////////////////
