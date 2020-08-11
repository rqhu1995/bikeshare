from sko.GA import *
from genetic_algorithm.config_reader import Config


class GeneticAlg:

    def __init__(self, fitness_fcn, lower_bound, upper_bound,
                 population_size, population_dimension):
        self.population_size = population_size
        self.population_dimension = population_dimension
        self.fitness_fcn = fitness_fcn
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def run_ga(self, initial_pop=0, mutation_pub=0.01,
               select_method=selection.selection_tournament_faster,
               rank_method=ranking.ranking_linear,
               crossover_method=crossover.crossover_2point,
               mutation_method=mutation.mutation):
        # user define
        lb = self.lower_bound
        ub = self.upper_bound
        crossover_method_functor = crossover_method
        select_method_functor = select_method
        ranking_method_functor = rank_method
        mutation_method_functor = mutation_method
        iteration = Config.read_config(['genetic_parameter', 'iteration_time'])
        mutation_prob = mutation_pub

        ga = GA(func=self.fitness_fcn, size_pop=self.population_size, n_dim=self.population_dimension,
                max_iter=iteration,
                prob_mut=mutation_prob,
                lb=lb, ub=ub,
                precision=[1] * self.population_dimension)
        if initial_pop is not None:
            ga.Chrom = initial_pop

        ga.register(operator_name='crossover', operator=crossover_method_functor). \
            register(operator_name='selection', operator=select_method_functor). \
            register(operator_name='ranking', operator=ranking_method_functor). \
            register(operator_name='mutation', operator=mutation_method_functor)

        best_x, best_y = ga.run()
        return best_x, best_y, ga
