import numpy as np
import random

class Game2048:
    """
    Implementation of the 2048 game.

    The game is played on a 4x4 grid where the player can move the tiles in four directions:

    - 'W' (up)
    - 'A' (left)
    - 'S' (down)
    - 'D' (right)

    Author: Filip Hajduch
    """
    def __init__(self):
        """
        Initialize the game board and score. Adds two tiles to start.
        """
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        """
        Add a new tile (2 or 4) to a random empty position on the board.
        """
        empty_positions = list(zip(*np.where(self.board == 0)))
        if empty_positions:
            row, col = random.choice(empty_positions)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """
        Shift all numbers in a row to the left, removing spaces.

        Args:
            row (numpy.ndarray): The row to be compressed.

        Returns:
            numpy.ndarray: The compressed row.
        """
        new_row = row[row != 0]
        new_row = np.concatenate((new_row, np.zeros(4 - len(new_row), dtype=int)))
        return new_row

    def merge(self, row):
        """
        Merge identical values in a row and update the score.

        Args:
            row (numpy.ndarray): The row to be merged.

        Returns:
            numpy.ndarray: The merged row.
        """
        for i in range(3):
            if row[i] == row[i+1] and row[i] != 0:
                row[i] *= 2
                self.score += row[i]
                row[i+1] = 0
        return row

    def move_left(self):
        """
        Move numbers left in the matrix and merge identical values.
        """
        for i in range(4):
            self.board[i] = self.compress(self.board[i])
            self.board[i] = self.merge(self.board[i])
            self.board[i] = self.compress(self.board[i])

    def move_right(self):
        """
        Move numbers right in the matrix and merge identical values.
        """
        for i in range(4):
            self.board[i] = self.compress(self.board[i][::-1])[::-1]
            self.board[i] = self.merge(self.board[i][::-1])[::-1]
            self.board[i] = self.compress(self.board[i][::-1])[::-1]

    def move_up(self):
        """
        Move numbers up in the matrix and merge identical values.
        """
        self.board = self.board.T
        self.move_left()
        self.board = self.board.T

    def move_down(self):
        """
        Move numbers down in the matrix and merge identical values.
        """
        self.board = self.board.T
        self.move_right()
        self.board = self.board.T

    def is_game_over(self):
        """
        Check if the game is over (no more valid moves).

        Returns:
            bool: True if the game is over, False otherwise.
        """
        if np.any(self.board == 0):
            return False
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1] or self.board[j, i] == self.board[j + 1, i]:
                    return False
        return True

    def print_board(self):
        """
        Print the current game board and score.
        """
        print(f"Score: {self.score}\n")
        for row in self.board:
            print(" | ".join(f"{num:4}" if num else "   ." for num in row))
        print("\nUse W (up), A (left), S (down), D (right) to move.")

    def play_turn(self, move):
        """
        Execute a move based on player input.

        Args:
            move (str): The move direction ('W', 'A', 'S', 'D').

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        if move == 'A':
            self.move_left()
        elif move == 'D':
            self.move_right()
        elif move == 'W':
            self.move_up()
        elif move == 'S':
            self.move_down()
        else:
            return False

        self.add_new_tile()
        return True

    def run(self):
        """
        Start the main game loop.
        """
        while True:
            self.print_board()
            move = input("Move (WASD): ").strip().upper()

            if move in ('W', 'A', 'S', 'D'):
                if not self.play_turn(move):
                    continue
            else:
                print("Invalid move, try again.")
                continue

            if self.is_game_over():
                self.print_board()
                print("Game Over! Score:", self.score)
                break
