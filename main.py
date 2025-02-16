from game_solvers import RandomSolver, MonteCarloSolver, HeuristicSolver

if __name__ == "__main__":

    random_solver = RandomSolver()
    montecarlo_solver = MonteCarloSolver()
    heuristic_solver = HeuristicSolver()


    random_solver.run()
    montecarlo_solver.run()
    heuristic_solver.run()
