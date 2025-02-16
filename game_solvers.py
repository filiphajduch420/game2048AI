import random
import numpy as np
import time
from game import Game2048


class StatisticsLogger:
    """
    Class for logging and saving statistics of AI solvers.
    """

    def __init__(self, solver_name, num_games):
        """
        Initialize the logger.

        Args:
            solver_name (str): Name of the solver being used.
            num_games (int): Number of games played.
        """
        self.solver_name = solver_name
        self.num_games = num_games
        self.scores = []
        self.max_tiles = []
        self.max_tile_games = []
        self.score_games = []
        self.move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        self.total_moves_per_game = []
        self.wins = 0
        self.execution_time = 0

    def record_game(self, game_number, score, max_tile, move_counts, total_moves):
        """
        Store statistics for a single game.

        Args:
            game_number (int): The number of the game.
            score (int): The final score of the game.
            max_tile (int): The highest tile achieved in the game.
            move_counts (dict): Number of moves per direction.
            total_moves (int): Total number of moves in the game.
        """
        self.scores.append(score)
        self.score_games.append(game_number)
        self.max_tiles.append(max_tile)
        self.max_tile_games.append(game_number)
        self.total_moves_per_game.append(total_moves)

        for key in self.move_counts:
            self.move_counts[key] += move_counts[key]

        if 2048 in self.max_tiles:
            self.wins += 1

    def log_results(self):
        """
        Display statistics from the games.
        """
        max_tile = max(self.max_tiles)
        max_tile_game = self.max_tile_games[self.max_tiles.index(max_tile)]
        best_score = max(self.scores)
        best_score_game = self.score_games[self.scores.index(best_score)]
        worst_score = min(self.scores)
        worst_score_game = self.score_games[self.scores.index(worst_score)]

        print(f"\n===== {self.solver_name} Solver Statistics =====")
        print(f"Number of games: {self.num_games}")
        print(f"Wins (reaching 2048): {self.wins}/{self.num_games}")
        print(f"Best score: {best_score} (game {best_score_game})")
        print(f"Worst score: {worst_score} (game {worst_score_game})")
        print(f"Average score: {sum(self.scores) / self.num_games:.2f}")
        print(f"Highest tile achieved: {max_tile} (game {max_tile_game})")
        print(f"Average number of moves per game: {sum(self.total_moves_per_game) / self.num_games:.2f}")
        print(f"Total execution time: {self.execution_time:.2f} seconds")

        for move, count in self.move_counts.items():
            print(f"Average moves {move}: {count / self.num_games:.2f}")

        print("===================================")

    def save_results_to_readme(self, filename="README.md"):
        """
        Append solver statistics to the README file instead of overwriting it.
        """
        max_tile = max(self.max_tiles)
        max_tile_game = self.max_tile_games[self.max_tiles.index(max_tile)]
        best_score = max(self.scores)
        best_score_game = self.score_games[self.scores.index(best_score)]
        worst_score = min(self.scores)
        worst_score_game = self.score_games[self.scores.index(worst_score)]

        # Read existing content of README.md
        try:
            with open(filename, "r") as file:
                existing_content = file.readlines()
        except FileNotFoundError:
            existing_content = []

        # Remove previous entry for this solver (if exists)
        start_marker = f"## {self.solver_name} Solver Results\n"
        end_marker = "## "  # Any other solver section starts with this
        new_content = []
        skip = False

        for line in existing_content:
            if line.startswith(start_marker):
                skip = True  # Start skipping previous solver's data
            elif skip and line.startswith(end_marker):
                skip = False  # Stop skipping when another solver's section starts
                new_content.append(line)
            elif not skip:
                new_content.append(line)

        # Append new results
        new_results = [
            f"## {self.solver_name} Solver Results\n",
            f"- **Number of games:** {self.num_games}\n",
            f"- **Wins (reaching 2048):** {self.wins}/{self.num_games}\n",
            f"- **Best score:** {best_score} (game {best_score_game})\n",
            f"- **Worst score:** {worst_score} (game {worst_score_game})\n",
            f"- **Average score:** {sum(self.scores) / self.num_games:.2f}\n",
            f"- **Highest tile achieved:** {max_tile} (game {max_tile_game})\n",
            f"- **Average number of moves per game:** {sum(self.total_moves_per_game) / self.num_games:.2f}\n",
            f"- **Total execution time:** {self.execution_time:.2f} seconds\n",
            "\n### Move Averages:\n",
        ]

        for move, count in self.move_counts.items():
            new_results.append(f"- **{move}:** {count / self.num_games:.2f} moves per game\n")

        new_results.append("\n")

        # Write updated content back
        with open(filename, "w") as file:
            file.writelines(new_content)
            file.writelines(new_results)


class RandomSolver:
    """
    Random solver for the 2048 game.
    """

    def __init__(self, num_games=30):
        self.logger = StatisticsLogger(solver_name="RandomSolver", num_games=num_games)
        self.num_games = num_games
        self.moves = ['W', 'A', 'S', 'D']

    def solve_one_game(self, game_number):
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        total_moves = 0

        while not game.is_game_over():
            move = random.choice(self.moves)
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()

class MonteCarloSolver:
    """
    Monte Carlo solver for the 2048 game.
    Simulates multiple games for each move and chooses the best move based on average score.
    """

    def __init__(self, num_games=30, simulations_per_move=10, max_simulation_depth=15):
        self.logger = StatisticsLogger(solver_name="MonteCarloSolver", num_games=num_games)
        self.num_games = num_games
        self.simulations_per_move = simulations_per_move
        self.max_simulation_depth = max_simulation_depth
        self.moves = ['W', 'A', 'S', 'D']

    def simulate_game(self, game, depth):
        """
        Play a simulated game from the current state for a fixed number of moves.
        """
        temp_game = Game2048()
        temp_game.board = np.copy(game.board)
        temp_game.score = game.score

        for _ in range(depth):
            if temp_game.is_game_over():
                break
            move = random.choice(self.moves)
            temp_game.play_turn(move)

        return temp_game.score - game.score

    def choose_best_move(self, game):
        """
        Simulate multiple games for each possible move and choose the best one.
        """
        best_move = None
        best_average_score = -1
        invalid_moves = set()

        for move in self.moves:
            temp_game = Game2048()
            temp_game.board = np.copy(game.board)
            temp_game.score = game.score

            if not temp_game.play_turn(move):
                invalid_moves.add(move)
                continue

            if np.array_equal(temp_game.board, game.board):
                invalid_moves.add(move)
                continue

            total_score = 0
            for _ in range(self.simulations_per_move):
                total_score += self.simulate_game(temp_game, self.max_simulation_depth)

            avg_score = total_score / self.simulations_per_move

            if avg_score > best_average_score:
                best_average_score = avg_score
                best_move = move

        if best_move is None:
            valid_moves = [move for move in self.moves if move not in invalid_moves]
            best_move = random.choice(valid_moves) if valid_moves else random.choice(self.moves)

        return best_move

    def solve_one_game(self, game_number):
        """
        Solve a single game using Monte Carlo simulations.
        """
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        total_moves = 0

        while not game.is_game_over():
            move = self.choose_best_move(game)
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        """
        Run multiple games and log statistics.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()

class HeuristicSolver:
    """
    Heuristic solver for 2048 game.
    Uses predefined rules to make optimal decisions.
    """

    def __init__(self, num_games=30):
        self.logger = StatisticsLogger(solver_name="HeuristicSolver", num_games=num_games)
        self.num_games = num_games
        self.moves = ['W', 'A', 'S', 'D']

    def evaluate_board(self, board):
        """
        Improved heuristic function to evaluate board strength.
        """
        score = 0

        # Prioritize empty spaces (fewer empty spaces = worse board)
        empty_spaces = np.count_nonzero(board == 0)
        score += empty_spaces * 10  # Weight of empty spaces

        # Prefer highest tile in the top-left corner
        highest_tile = np.max(board)
        if board[0, 0] == highest_tile:
            score += highest_tile * 10  # Large bonus for keeping high tile in corner

        # Monotonicity
        for row in board:
            for i in range(3):
                if row[i] > row[i + 1]:
                    score += 1

        for col in board.T:
            for i in range(3):
                if col[i] > col[i + 1]:
                    score += 1

        # Smoothness
        for i in range(4):
            for j in range(3):
                score -= abs(board[i, j] - board[i, j + 1])
                score -= abs(board[j, i] - board[j + 1, i])

        return score

    def choose_best_move(self, game):
        """
        Choose the best move based on heuristic evaluation.
        """
        best_move = None
        best_score = -1
        invalid_moves = set()

        for move in self.moves:
            temp_game = Game2048()
            temp_game.board = np.copy(game.board)

            if not temp_game.play_turn(move):
                invalid_moves.add(move)
                continue  # Skip invalid moves

            if np.array_equal(temp_game.board, game.board):
                invalid_moves.add(move)
                continue

            score = self.evaluate_board(temp_game.board)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            valid_moves = [move for move in self.moves if move not in invalid_moves]
            best_move = random.choice(valid_moves) if valid_moves else random.choice(self.moves)

        return best_move

    def solve_one_game(self, game_number):
        """
        Solve a single game using heuristic evaluation.
        """
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        total_moves = 0

        while not game.is_game_over():
            move = self.choose_best_move(game)
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        """
        Run multiple games and log statistics.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()