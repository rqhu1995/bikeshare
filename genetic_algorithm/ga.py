import conf_reader as conf
from database import cluster_info
from sko.GA import GA, selection
from static_bikeshare.rebalancing_route import customized_mutation, customized_crossover, fitness_function_rebalancing, \
    generate_feasible_population


def config_ga(fitness_fun, lower_bound=None, upper_bound=None, constraint_ueq=None, constraint_eq=None):
    iteration_time = int(conf.get_val('ga_param', 'iteration_time'))
    population_size = int(conf.get_val('ga_param', 'population_size'))
    crossover_type = conf.get_val('ga_param', 'crossover_type')
    mutation_type = conf.get_val('ga_param', 'mutation_type')
    station_count = int(conf.get_val('model_param', 'station_count'))

    lb = lower_bound
    ub = upper_bound
    constraint_eq = constraint_eq
    ga = GA(func=fitness_fun, size_pop=population_size, n_dim=station_count, max_iter=iteration_time)
    if crossover_type == 'customized':
        ga.register(operator_name='crossover', operator=customized_crossover)
    if mutation_type == 'customized':
        ga.register(operator_name='mutation', operator=customized_mutation)
    # ga.register(operator_name='selection', operator=selection.selection_tournament)
    return ga


def ga_begin(ga):
    best_x, best_y = ga.run()
    print(best_x, best_y)
    return best_x, best_y, ga


all_stations = [station['station_id'] for station in cluster_info[3]]
ga_instance = config_ga(fitness_function_rebalancing)
ga_instance.Chrom = generate_feasible_population(all_stations, int(conf.get_val('model_param', 'truck_count')),
                                                 int(conf.get_val('ga_param', 'population_size')))
print(ga_instance.Chrom.shape)
ga_begin(ga_instance)