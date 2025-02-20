from game_solvers import RandomSolver, MonteCarloSolver, HeuristicSolver, StatisticsLogger

if __name__ == "__main__":
    random_solver = RandomSolver()
    montecarlo_solver = MonteCarloSolver()
    heuristic_solver = HeuristicSolver()

    random_solver.run()
    montecarlo_solver.run()
    heuristic_solver.run()

    random_solver.logger.generate_graph(filename="graphs/random_solver_results.png")
    montecarlo_solver.logger.generate_graph(filename="graphs/montecarlo_solver_results.png")
    heuristic_solver.logger.generate_graph(filename="graphs/heuristic_solver_results.png")

    StatisticsLogger.generate_comparison_graphs([random_solver, montecarlo_solver, heuristic_solver])