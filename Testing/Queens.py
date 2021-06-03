import random
import math
from multiprocessing import Process
import time
N = 100
def cost(board):
    """Calculate Fitness/Cost of given board.

    Finds all clashes with each queen left to right.
    Using reletive positioning to find if any queens are on diagonals
    # and checks if they lay on the same horizontal.

    Args:
        board (int)[]: A board represented as each index as column and value as row position.

    Returns:
        (int): A count of all the queens that are clashing with another

    Examples:
        An Example showing for a board size N = 8 getting the cost.

        >>> print(cost([x for x in range(8)]))
        28
    """
    cost = 0 # Stores the total cost to return.
    for i in range(len(board)-1): # For all queens ignoring last since there is no queen after it
        diff_up = board[i] - i
        diff_down = board[i] + i
        for j in range(i+1,len(board)): # Compares all other queens and checks their reletive position and row
            # These three checks, check if any queen to the right
            # lie on the same diagonal +/-, or on the same row.
            if diff_up == board[j] - j:
                cost += 1
            elif diff_down == board[j] + j:
                cost += 1
            if board[i] == board[j]:
                cost += 1
    return cost

class Board:
    """

    """
    def __init__(self, board):
        self.board = board
        self.cost = cost(self.board)

    def __str__(self):
        output = ""
        for i in range(N):
            for j in range(N):
                if i == self.board[j]:
                    output += "♕ "
                    #"♕ "
                else:
                    output += "# "
                    # new_board = list(self.board)
                    # new_board[j] = i
                    # output += "{0:3d} ".format(cost(new_board))
            output += "\n"
        output += "Cost = " + str(self.cost) + "\n"
        output += "Board = [" + ", ".join([str(x) for x in self.board]) + "]"
        return output

    def __lt__(self, other):
        return self.cost < other.cost


def randomise_solve(board):
    count = 1
    while board.cost > 0:
        board = Board([random.randint(0,N-1) for x in range(N)])
        count += 1
    return board, count

def generate_initial_population(pool_size):
    boards = []
    for i in range(pool_size):
        x = Board([random.randint(0,N-1) for x in range(N)])
        boards.append(x)
    return boards

def genetic_solve():
    pool_size = 200
    selection_factor = 4
    mutate_chance = 0.8
    max_generations = 10000

    start_time = time.time()

    generations = 1
    population = generate_initial_population(pool_size)
    population.sort()
    # print(population[0])
    min_cost = population[0].cost
    while population[0].cost > 0 and generations < max_generations:
        new_population = []
        for i in range(int(pool_size/2)):
            random_parent_one = population[int(pool_size*random.random()**selection_factor)]
            random_parent_two = population[int(pool_size*random.random()**selection_factor)]
            random_splice = random.randint(1,N-2)
            child_one_board = random_parent_one.board[0:random_splice] + random_parent_two.board[random_splice:N]
            if random.random() < mutate_chance:
                child_one_board[random.randint(0,N-1)] = random.randint(0,N-1)
            child_two_board = random_parent_two.board[0:random_splice] + random_parent_one.board[random_splice:N]
            if random.random() < mutate_chance:
                child_two_board[random.randint(0,N-1)] = random.randint(0,N-1)

            child_one = Board(child_one_board)
            child_two = Board(child_two_board)

            new_population.append(child_one)
            new_population.append(child_two)
        population = new_population
        population.sort()
        generations += 1
        # if population[0].cost < min_cost:
        #     print("Generation:", generations, "\n\tCost=", population[0].cost)
        #     min_cost = population[0].cost
    print("_________Genetic______________took: ", time.time() - start_time)
    print(population[0])
    print("Generations = ", generations)

def neighbours_gen(board):
    neighbours = []
    for i in range(N):
        for j in range(N):
            if (i,j) != (i,board[i]):
                neighbours.append((i,j))
    return neighbours

def get_best_neighbours(board):
    neighbours = []
    X = Board(board)
    min = X.cost - 1
    for i in range(N):
        for j in range(N):
            if (i,j) != (i,board[i]):
                new_board = list(X.board)
                new_board[i] = j
                next_X = Board(new_board)
                if next_X.cost < min:
                    neighbours = []
                    neighbours.append(next_X)
                    min = next_X.cost
                elif next_X.cost == min:
                    neighbours.append(next_X)
    return neighbours

def steep_hill_climb_solve():
    start_time = time.time()
    X = Board([random.randint(0,N-1) for x in range(N)])
    neighbours = get_best_neighbours(X.board)
    # print(X)
    while X.cost > 0 and len(neighbours) > 0:
        neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
        X = neighbour
        neighbours = get_best_neighbours(X.board)
    print("______Steep Hill Climb____________took: ", time.time() - start_time)
    print(X)

def hill_climb_solve():
    start_time = time.time()
    X = Board([random.randint(0,N-1) for x in range(N)])
    neighbours = neighbours_gen(X.board)
    # print(X)
    while X.cost > 0 and len(neighbours) > 0:
        neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
        new_board = list(X.board)
        new_board[neighbour[0]] = neighbour[1]
        next_X = Board(new_board)
        if next_X.cost < X.cost:
            X = next_X
            neighbours = neighbours_gen(X.board)
    print("_________Hill Climb______________took: ", time.time() - start_time)
    print(X)

def restart_steep_hill_climb_solve():
    start_time = time.time()
    X = Board([random.randint(0,N-1) for x in range(N)])
    neighbours = get_best_neighbours(X.board)
    # print(X)
    while X.cost > 0:
        while len(neighbours) > 0:
            neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
            X = neighbour
            neighbours = get_best_neighbours(X.board)
        if X.cost > 0:
            X = Board([random.randint(0,N-1) for x in range(N)])
            neighbours = get_best_neighbours(X.board)
    print("_______Steep Restart__________took: ", time.time() - start_time)
    print(X)

def restart_hill_climb_solve():
    start_time = time.time()
    X = Board([random.randint(0,N-1) for x in range(N)])
    neighbours = neighbours_gen(X.board)
    # print(X)
    while X.cost > 0:
        X = Board([random.randint(0,N-1) for x in range(N)])
        neighbours = neighbours_gen(X.board)
        while len(neighbours) > 0:
            neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
            new_board = list(X.board)
            new_board[neighbour[0]] = neighbour[1]
            next_X = Board(new_board)
            if next_X.cost < X.cost:
                X = next_X
                neighbours = neighbours_gen(X.board)
    print("_________Restart__________took: ", time.time() - start_time)
    print(X)

def simulated_annealing_solve(T, alpha, K):
    T_min = 0.000001
    X = Board([random.randint(0,N-1) for x in range(N)])
    neighbours = neighbours_gen(X.board)
    start_time = time.time()
    while X.cost > 0 and T > T_min and len(neighbours) > 0:
        for k in range(K):
            neighbour = neighbours.pop(random.randint(0, len(neighbours)-1))
            new_board = list(X.board)
            new_board[neighbour[0]] = neighbour[1]
            next_X = Board(new_board)
            if next_X.cost <= X.cost:
                X = next_X
                neighbours = neighbours_gen(X.board)
            elif math.e**(-1*(next_X.cost-X.cost)/T) > random.uniform(0, 1):
                X = next_X
                neighbours = neighbours_gen(X.board)
        T = T*alpha
    print("_________Simulated Annealing_______took: ", time.time() - start_time)
    print(X)

# def sa():
#     T = 1.0
#     T_min = 0.000000000001
#     alpha = 0.99
#     K = 100
#     start_time = time.time()
#     board = [x for x in range(N)]
#     random.shuffle(board)
#     min = cost(board)
#     # print(X)
#     while min > 0 and T > T_min:
#         for k in range(K):
#             pos = random.sample(range(N), 2)
#             new_board = list(board)
#             new_board[pos[0]], new_board[pos[1]] = new_board[pos[1]], new_board[pos[0]]
#             next_X = cost(new_board)
#             P = 0.0
#             try:
#                 P = math.e**(-1*(next_X-min)/T)
#             except:
#                 P = 0.0
#             if P > random.uniform(0, 1):
#                 min = next_X
#                 print(min)
#                 board = new_board
#         T = T*alpha
#     print("_________Simulated Annealing_______took: ", time.time() - start_time)
#     print(T)
#     print(board)

# def sa():
#     T = 1.0
#     T_min = 0.000000000001
#     alpha = 0.99
#     K = 100
#     start_time = time.time()
#     board = [x for x in range(N)]
#     random.shuffle(board)
#     X = Board(board)
#     min = X.cost
#     # print(X)
#     while X.cost > 0 and T > T_min:
#         for k in range(K):
#             pos = random.sample(range(N), 2)
#             new_board = list(X.board)
#             new_board[pos[0]], new_board[pos[1]] = new_board[pos[1]], new_board[pos[0]]
#             next_X = Board(new_board)
#             P = 0.0
#             try:
#                 P = math.e**(-1*(next_X.cost-X.cost)/T)
#             except:
#                 P = 0.0
#             if P > random.uniform(0, 1):
#                 if next_X.cost < min:
#                     print(next_X.cost)
#                     min = next_X.cost
#                 X = next_X
#         T = T*alpha
#     print("_________Simulated Annealing_______took: ", time.time() - start_time)
#     print(T)
#     print(X)

def poor_man_solution():
    start_time = time.time()
    board = [x for x in range(N)]
    if N%2==0:
        if N%6!=2:
            for m in range(int(N/2)):
                board[m] = 2*m+1
                board[int(N/2+m)] = 2*m-1+1
        elif N%6!=0:
            for m in range(int(N/2)):
                board[m] = 1+(2*(m-1)+N/2-1)%N+1
                board[N+1-m] = N-(2*(m-1)+n/2-1)%N+1
    else:
        n = N-1
        if n%6!=2:
            for m in range(int(n/2)):
                board[m] = 2*m+1
                board[int(n/2+m)] = 2*m-1+1
        elif n%6!=0:
            for m in range(int(n/2)):
                board[m] = 1+(2*(m-1)+n/2-1)%n
                board[n-m] = n-(2*(m-1)+n/2-1)%n
        board[N-1] = 0
    # X = Board(board)

    # print(board)
    print("Cost:", 0)
    print("Took:", time.time() - start_time)
    print("N:", N)


if __name__ == "__main__":
    # ga = Process(target=genetic_solve)
    # ga.start()
    # sa = Process(target=simulated_annealing_solve)
    # sa.start()
    # hc = Process(target=hill_climb_solve)
    # hc.start()
    # rs = Process(target=restart_hill_climb_solve)
    # rs.start()
    # shc = Process(target=steep_hill_climb_solve)
    # shc.start()
    # srs = Process(target=restart_steep_hill_climb_solve)
    # srs.start()
    # ga.join()
    # sa.join()
    # hc.join()
    # rs.join()
    # shc.join()
    # srs.join()
    # genetic_solve()
    simulated_annealing_solve()
    # hill_climb_solve()
    # restart_hill_climb_solve()
    # sa()
    # poor_man_solution()
