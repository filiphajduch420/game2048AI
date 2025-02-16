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
