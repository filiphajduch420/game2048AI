import random
import numpy as np
import time
import matplotlib.pyplot as plt
from game import Game2048


class StatisticsLogger:
    """
    Class for logging and saving statistics of AI solvers.

    This class records game statistics, calculates performance metrics,
    and saves results to a file for analysis.
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

    def generate_graph(self, filename="results_graph.png"):
        """
        Generate a graph of the achieved scores for each solver.

        Args:
            filename (str): The filename to save the graph as.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.score_games, self.scores, label=self.solver_name, marker='o')

        plt.xlabel('Measurement')
        plt.ylabel('Score')
        plt.title('Achieved Scores of Solvers')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def generate_comparison_graphs(solvers):
        """
        Generate comparison graphs for different solvers.

        This function creates graphs comparing the average scores, win rates, and execution times of different solvers.

        Args:
            solvers (list): A list of solver instances.
        """
        solver_names = [solver.logger.solver_name for solver in solvers]
        average_scores = [sum(solver.logger.scores) / solver.logger.num_games for solver in solvers]
        win_rates = [solver.logger.wins / solver.logger.num_games for solver in solvers]
        execution_times = [solver.logger.execution_time for solver in solvers]

        # Average Score Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(solver_names, average_scores, color='skyblue')
        plt.xlabel('Solver')
        plt.ylabel('Average Score')
        plt.title('Average Score Comparison')
        plt.savefig('graphs/average_score_comparison.png')
        plt.close()

        # Win Rate Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(solver_names, win_rates, color='lightgreen')
        plt.xlabel('Solver')
        plt.ylabel('Win Rate')
        plt.title('Win Rate Comparison')
        plt.savefig('graphs/win_rate_comparison.png')
        plt.close()

        # Execution Time Comparison
        plt.figure(figsize=(10, 6))
        plt.bar(solver_names, execution_times, color='salmon')
        plt.xlabel('Solver')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.savefig('graphs/execution_time_comparison.png')
        plt.close()

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
        Display statistics from the games in the console.
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
        start_marker = f"## {self.solver_name} Results\n"
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

    This solver selects moves randomly without any strategy.
    It serves as a baseline for comparison with more advanced solvers.
    """

    def __init__(self, num_games=30):
        """
        Initialize the Random Solver.

        Args:
            num_games (int): The number of games to be played. Default is 30.
        """
        self.logger = StatisticsLogger(solver_name="RandomSolver", num_games=num_games)
        self.num_games = num_games
        self.moves = ['W', 'A', 'S', 'D']  # Possible moves (Up, Left, Down, Right)

    def solve_one_game(self, game_number):
        """
        Play one game using random moves.

        Args:
            game_number (int): The index of the game being played.
        """
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}  # Track move counts
        total_moves = 0

        while not game.is_game_over():
            move = random.choice(self.moves)  # Choose a random move
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        # Log the game's results
        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        """
        Run multiple games using the Random Solver and log results.

        This function plays multiple games, records statistics, and saves the results.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()

class MonteCarloSolver:
    """
    Monte Carlo solver for the 2048 game.

    This solver uses Monte Carlo simulations to determine the best move.
    For each possible move, it simulates multiple random games from the resulting state
    and selects the move that leads to the highest average score.

    The solver balances performance and accuracy by adjusting the number of simulations per move
    and the maximum depth of each simulation.
    """

    def __init__(self, num_games=30, simulations_per_move=10, max_simulation_depth=15):
        """
        Initialize the Monte Carlo Solver.

        Args:
            num_games (int): The number of games to be played. Default is 30.
            simulations_per_move (int): The number of simulations for each move. Default is 10.
            max_simulation_depth (int): The maximum number of moves simulated in a single game. Default is 15.
        """
        self.logger = StatisticsLogger(solver_name="MonteCarloSolver", num_games=num_games)
        self.num_games = num_games  # Total number of games to be played
        self.simulations_per_move = simulations_per_move  # Number of simulations for each move
        self.max_simulation_depth = max_simulation_depth  # Maximum number of moves in a single simulation
        self.moves = ['W', 'A', 'S', 'D']  # Possible moves (Up, Left, Down, Right)

    def simulate_game(self, game, depth):
        """
        Play a simulated game from the current state for a fixed number of moves.

        This function creates a copy of the game state and plays random moves until the maximum depth is reached
        or the game is over. The difference in score between the initial and final states is returned.

        Args:
            game (Game2048): The current game state to simulate from.
            depth (int): The maximum number of moves to simulate.

        Returns:
            int: The score gained during the simulation.
        """
        temp_game = Game2048()
        temp_game.board = np.copy(game.board)  # Copy the board state
        temp_game.score = game.score  # Copy the current score

        for _ in range(depth):
            if temp_game.is_game_over():
                break
            move = random.choice(self.moves)  # Randomly choose a move
            temp_game.play_turn(move)

        return temp_game.score - game.score  # Return the score gained during the simulation

    def choose_best_move(self, game):
        """
        Simulate multiple games for each possible move and choose the best one.

        The solver evaluates each possible move by performing multiple simulations
        and selecting the move that results in the highest average score.

        Args:
            game (Game2048): The current game state.

        Returns:
            str: The best move ('W', 'A', 'S', or 'D').
        """
        best_move = None
        best_average_score = -1
        invalid_moves = set()

        for move in self.moves:
            temp_game = Game2048()
            temp_game.board = np.copy(game.board)
            temp_game.score = game.score

            # Try playing the move on a temporary game state
            if not temp_game.play_turn(move):
                invalid_moves.add(move)
                continue  # Skip invalid moves

            if np.array_equal(temp_game.board, game.board):  # If the move results in no change
                invalid_moves.add(move)
                continue

            # Run multiple simulations from this move and calculate the average score
            total_score = 0
            for _ in range(self.simulations_per_move):
                total_score += self.simulate_game(temp_game, self.max_simulation_depth)

            avg_score = total_score / self.simulations_per_move  # Compute the average score

            # Update the best move if this one performs better
            if avg_score > best_average_score:
                best_average_score = avg_score
                best_move = move

        # If no valid move was found, pick a random valid move
        if best_move is None:
            valid_moves = [move for move in self.moves if move not in invalid_moves]
            best_move = random.choice(valid_moves) if valid_moves else random.choice(self.moves)

        return best_move

    def solve_one_game(self, game_number):
        """
        Solve a single game using Monte Carlo simulations.

        This function runs a single game loop where moves are chosen based on
        the Monte Carlo evaluation until the game reaches a terminal state.

        Args:
            game_number (int): The index of the game being played.
        """
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}  # Track move counts
        total_moves = 0

        while not game.is_game_over():
            move = self.choose_best_move(game)  # Get the best move based on Monte Carlo simulations
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        # Log the game's results
        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        """
        Run multiple games using the Monte Carlo solver and log results.

        This function plays multiple games, records statistics, and saves the results.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()


class HeuristicSolver:
    """
    Heuristic solver for the 2048 game.

    This solver makes decisions based on predefined heuristics rather than simulations or brute-force search.
    It aims to maintain an optimal board structure by prioritizing empty spaces, positioning high-value tiles,
    and ensuring smooth and monotonic tile distribution.
    """

    def __init__(self, num_games=30):
        """
        Initialize the Heuristic Solver.

        Args:
            num_games (int): The number of games to be played. Default is 30.
        """
        self.logger = StatisticsLogger(solver_name="HeuristicSolver", num_games=num_games)
        self.num_games = num_games  # Total number of games to be played
        self.moves = ['W', 'A', 'S', 'D']  # Possible moves (Up, Left, Down, Right)

    def evaluate_board(self, board):
        """
        Heuristic function to evaluate the strength of a given board.

        The heuristic is based on the following factors:
        - Empty spaces: More empty spaces provide better flexibility for merging tiles.
        - Highest tile at the top-left corner: Encourages a structured board layout.
        - Monotonicity: Prefers a board where values decrease smoothly across rows and columns.
        - Smoothness: Penalizes large differences between adjacent tiles.

        Args:
            board (np.ndarray): The 4x4 game board.

        Returns:
            int: The computed heuristic score for the board.
        """
        score = 0

        # Prefer boards with more empty spaces (fewer tiles = more flexibility)
        empty_spaces = np.count_nonzero(board == 0)
        score += empty_spaces * 10  # Assign weight to empty spaces

        # Prefer keeping the highest tile at the top-left corner
        highest_tile = np.max(board)
        if board[0, 0] == highest_tile:  # Check if highest tile is in the top-left corner
            score += highest_tile * 10  # Large bonus for structured layout

        # Monotonicity: Prefer decreasing values across rows and columns
        for row in board:
            for i in range(3):
                if row[i] > row[i + 1]:  # Prefer left-to-right decreasing values
                    score += 1

        for col in board.T:  # Transpose to check columns as rows
            for i in range(3):
                if col[i] > col[i + 1]:  # Prefer top-to-bottom decreasing values
                    score += 1

        # Smoothness: Penalize large differences between adjacent tiles
        for i in range(4):
            for j in range(3):
                score -= abs(board[i, j] - board[i, j + 1])  # Penalize differences in rows
                score -= abs(board[j, i] - board[j + 1, i])  # Penalize differences in columns

        return score

    def choose_best_move(self, game):
        """
        Choose the best move based on heuristic evaluation.

        The solver plays each possible move on a temporary board and evaluates
        the resulting state using the heuristic function.

        Args:
            game (Game2048): The current game state.

        Returns:
            str: The best move ('W', 'A', 'S', or 'D').
        """
        best_move = None
        best_score = -1
        invalid_moves = set()

        for move in self.moves:
            temp_game = Game2048()
            temp_game.board = np.copy(game.board)  # Copy the board state

            # Try playing the move on a temporary game state
            if not temp_game.play_turn(move):
                invalid_moves.add(move)
                continue  # Skip invalid moves

            # If the move results in no board change, it's useless
            if np.array_equal(temp_game.board, game.board):
                invalid_moves.add(move)
                continue

            # Evaluate the new board state
            score = self.evaluate_board(temp_game.board)
            if score > best_score:
                best_score = score
                best_move = move

        # If no valid move was found, pick a random valid move
        if best_move is None:
            valid_moves = [move for move in self.moves if move not in invalid_moves]
            best_move = random.choice(valid_moves) if valid_moves else random.choice(self.moves)

        return best_move

    def solve_one_game(self, game_number):
        """
        Solve a single game using heuristic evaluation.

        This function runs a single game loop where moves are chosen based on
        heuristic evaluation until the game reaches a terminal state.

        Args:
            game_number (int): The index of the game being played.
        """
        game = Game2048()
        move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}  # Track move counts
        total_moves = 0

        while not game.is_game_over():
            move = self.choose_best_move(game)  # Get the best move based on heuristics
            if game.play_turn(move):
                move_counts[move] += 1
                total_moves += 1

        # Log the game's results
        self.logger.record_game(game_number, game.score, np.max(game.board), move_counts, total_moves)

    def run(self):
        """
        Run multiple games using the heuristic solver and log results.

        This function plays multiple games, records statistics, and saves the results.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.logger.execution_time = time.time() - start_time
        self.logger.log_results()
        self.logger.save_results_to_readme()