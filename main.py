from game_solvers import RandomSolver, MonteCarloSolver, HeuristicSolver

if __name__ == "__main__":

    random_solver = RandomSolver()
    montecarlo_solver = MonteCarloSolver(num_games=30, simulations_per_move=10, max_simulation_depth=15)
    heuristic_solver = HeuristicSolver()


    random_solver.run()
    montecarlo_solver.run()
    heuristic_solver.run()


    random_solver.logger.generate_graph(filename="random_solver_results.png")
    montecarlo_solver.logger.generate_graph(filename="montecarlo_solver_results.png")
    heuristic_solver.logger.generate_graph(filename="heuristic_solver_results.png")