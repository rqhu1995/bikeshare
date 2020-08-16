import json
import time

import numpy as np

import conf_reader as conf
from database import cluster_info
from sko.GA import GA
from static_bikeshare.rebalancing_route import customized_mutation, customized_crossover, fitness_function_rebalancing, \
    generate_feasible_population

selected_cluster = int(conf.get_val('model_param', 'selected_cluster'))


def config_ga(fitness_fun, lower_bound=None, upper_bound=None, constraint_ueq=None, constraint_eq=None):
    iteration_time = int(conf.get_val('ga_param', 'iteration_time'))
    population_size = int(conf.get_val('ga_param', 'population_size'))
    crossover_type = conf.get_val('ga_param', 'crossover_type')
    mutation_type = conf.get_val('ga_param', 'mutation_type')
    station_count = int(conf.get_val('model_param', 'station_count'))
    truncate_per_stage = int(conf.get_val('model_param', 'truncate_per_stage'))

    lb = lower_bound
    ub = upper_bound
    constraint_eq = constraint_eq
    ga = GA(func=fitness_fun, size_pop=population_size, n_dim=truncate_per_stage, max_iter=iteration_time)
    if crossover_type == 'customized':
        ga.register(operator_name='crossover', operator=customized_crossover)
    if mutation_type == 'customized':
        ga.register(operator_name='mutation', operator=customized_mutation)
    return ga


def ga_begin(ga):
    t = time.time()
    best_x, best_y = ga.run()
    print('遗传算法运行时长：' + str(time.time() - t))
    print(best_x, best_y)
    return best_x, best_y, ga


all_stations = [station['station_id'] for station in cluster_info[int(conf.get_val('model_param', 'selected_cluster'))]]
ga_instance = config_ga(fitness_function_rebalancing)
ga_instance.Chrom = generate_feasible_population(all_stations, int(conf.get_val('model_param', 'truck_count')),
                                                 int(conf.get_val('ga_param', 'population_size')))
# print(ga_instance.Chrom.shape)
best_route = ga_begin(ga_instance)[0]
route_fp = open('resources/exp_result/route_info.json', encoding='utf-8')
all_routes = route_fp.readlines()
final_json = '['
for route in all_routes:
    final_json += route
final_json = final_json[:-2] + ']'
final_routes = json.loads(final_json)
route_fp.close()
best_record = None
for record in final_routes:
    if record['route_info'][0]['route'] + [0] + record['route_info'][1]['route'] == list(best_route):
        best_record = record
        break
best_route_fp = open('resources/exp_result/best_route.json', mode='a+', encoding='utf-8')
json.dump(best_record, best_route_fp)
best_route_fp.write('\n卡车1剩余车辆数：' +
                    str(30 - np.sum(best_record['route_info'][0]['actual_allocation'])) + '\n'
                    )
best_route_fp.write('卡车2剩余车辆数：' +
                    str(30 - np.sum(best_record['route_info'][1]['actual_allocation'])) + '\n'
                    )
best_route_fp.close()
