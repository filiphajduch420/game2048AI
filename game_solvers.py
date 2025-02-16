import random
import numpy as np
import time
from game import Game2048


class RandomSolver:
    """
    Solver that selects random moves in the 2048 game and logs statistics.
    """

    def __init__(self, num_games=30):
        """
        Initialize the solver with the number of games to be played.

        Args:
            num_games (int): Number of games to be played.
        """
        self.num_games = num_games
        self.moves = ['W', 'A', 'S', 'D']  # Possible moves

        # Statistics
        self.scores = []
        self.max_tiles = []
        self.max_tile_games = []
        self.score_games = []
        self.move_counts = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        self.total_moves_per_game = []
        self.wins = 0
        self.execution_time = 0

    def solve_one_game(self, game_number):
        """
        Play one game using a random solver and collect statistics.
        """
        game = Game2048()
        moves_count = {'W': 0, 'A': 0, 'S': 0, 'D': 0}
        total_moves = 0

        while not game.is_game_over():
            move = random.choice(self.moves)
            if game.play_turn(move):
                moves_count[move] += 1
                total_moves += 1

        self.scores.append(game.score)
        self.score_games.append(game_number)
        max_tile = np.max(game.board)
        self.max_tiles.append(max_tile)
        self.max_tile_games.append(game_number)
        self.total_moves_per_game.append(total_moves)

        for key in self.move_counts:
            self.move_counts[key] += moves_count[key]

        if 2048 in game.board:
            self.wins += 1

    def run(self):
        """
        Play multiple games and compute statistics, measuring execution time.
        """
        start_time = time.time()

        for game_number in range(1, self.num_games + 1):
            self.solve_one_game(game_number)

        self.execution_time = time.time() - start_time

        self.log_results()
        self.save_results_to_readme()

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

        print("\n===== Random Solver Statistics =====")
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
        Save statistics from the games to a README file.
        """
        max_tile = max(self.max_tiles)
        max_tile_game = self.max_tile_games[self.max_tiles.index(max_tile)]
        best_score = max(self.scores)
        best_score_game = self.score_games[self.scores.index(best_score)]
        worst_score = min(self.scores)
        worst_score_game = self.score_games[self.scores.index(worst_score)]

        with open(filename, "w") as file:
            file.write("# 2048 AI Solver - Random Strategy\n\n")
            file.write("## Latest Performance Results\n\n")
            file.write(f"- **Number of games:** {self.num_games}\n")
            file.write(f"- **Wins (reaching 2048):** {self.wins}/{self.num_games}\n")
            file.write(f"- **Best score:** {best_score} (game {best_score_game})\n")
            file.write(f"- **Worst score:** {worst_score} (game {worst_score_game})\n")
            file.write(f"- **Average score:** {sum(self.scores) / self.num_games:.2f}\n")
            file.write(f"- **Highest tile achieved:** {max_tile} (game {max_tile_game})\n")
            file.write(f"- **Average number of moves per game:** {sum(self.total_moves_per_game) / self.num_games:.2f}\n")
            file.write(f"- **Total execution time:** {self.execution_time:.2f} seconds\n\n")

            file.write("### Move Averages:\n")
            for move, count in self.move_counts.items():
                file.write(f"- **{move}:** {count / self.num_games:.2f} moves per game\n")

            file.write("\n_Last updated automatically after the last test run._\n")