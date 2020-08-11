import time

import matplotlib.pyplot as plt
from sko.GA import GA
from sko.operators import crossover

from genetic_algorithm.config_reader import Config as cfg
from genetic_algorithm.database import *


# station_count =
# station_count = int(cfg.read_config('station_count'))


def fitness_fun(x):
    """Definition of the fitness function
    :param x: the allocation for bicycles at the beginning of each hour
    :return:
    """
    # user's borrow count estimated
    rent_amount = np.zeros((3, station_count))
    return_amount = np.zeros((3, station_count))
    # bike available in this hour
    available = (x * 20)
    print(list(np.ceil(available).astype(int)))
    # print(available)
    # value of the objective function
    fval_rent_success = 0
    fval_return_fail = 0
    rent_demand = np.array(rent_6_9[:,range(50)])
    return_demand = np.array(return_6_9[:,range(50)])
    for current_stage in range(0, 3):
        for station_id in range(station_count):
            rent_amount[current_stage][station_id] = min(rent_demand[current_stage][station_id], available[station_id])
            return_amount[current_stage][station_id] = min(return_demand[current_stage][station_id],
                                                           max(0, max_capacity[0][station_id] - available[station_id]))
            available[station_id] = available[station_id] - rent_amount[current_stage][station_id] + \
                                    return_amount[current_stage][station_id]
            fval_rent_success += rent_amount[current_stage][station_id]
            fval_return_fail += max(0,
                                    return_demand[current_stage][station_id] - return_amount[current_stage][station_id])
            # print(available[station_id])
    obj = -fval_rent_success
    print(obj)
    # print(rent_amount)
    return obj


def config_ga():
    constraint_ueq = [
        lambda x:  sum(x)*20 - int(cfg.read_config(['bike_problem_config', 'bike_total_count']))
    ]
    lb = [0] * station_count
    ub = []
    for i in range(50):
        ub.append(max_capacity[0][i]/20)
    # print(ub)
    iteration = int(cfg.read_config(['genetic_parameter', 'iteration_time']))
    ga = GA(func=fitness_fun, size_pop=50, n_dim=station_count, max_iter=iteration, prob_mut=0.01, lb=lb, ub=ub,
            constraint_ueq=constraint_ueq)
    if cfg.read_config(['genetic_parameter', 'crossover_type']) == 'None':
        ga.register(operator_name='crossover', operator=crossover.crossover_1point)
    # TODO: 变异和排序的策略配置（暂时不考虑）
    return ga


def ga_begin(ga):
    best_x, best_y = ga.run()
    print(best_x, best_y)
    return best_x, best_y, ga


#
def ga_plot(ga, bike_allocation):
    # print(ga.generation_best_Y)
    fig, ax = plt.subplots()
    generation = []
    best = []
    valid_gen = 0
    # print('best_Y:')
    # print(ga.generation_best_X)
    for alloc in ga.generation_best_X:
        best.append(-fitness_fun(alloc))
        print(-fitness_fun(alloc))
    # for gen in ga.generation_best_Y:
    #     if gen < 0:
    #         best.append(gen)
    #         generation.append(valid_gen)
    #         valid_gen = valid_gen + 1
    # print(generation)
    # print(best)
    plt.xlim((0, cfg.read_config(['genetic_parameter', 'iteration_time'])))
    plt.ylim((8000, 10000))
    plt.xlabel('iteration time')
    plt.ylabel('fitness value')
    my_x_ticks = np.arange(0, iteration_time, 10)
    my_y_ticks = np.arange(8000, 10000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    # f_val = -fitness_fun(bike_allocation)
    # print(best)
    # idx = best.index(-f_val)
    # max_cord = (idx, int(-fitness_fun(bike_allocation)))
    # max_cord_shift = (idx + 2, int(-fitness_fun(bike_allocation)) + 10)
    ax.plot(range(0, iteration_time), np.array(best), color='red')

    # plt.vlines(idx, 300, int(f_val), 'black', '--', label='example')
    # plt.hlines(int(f_val), 0, idx, 'black', '--', label='example')
    # plt.annotate("(" + str(idx) + ", " + str(-fitness_fun(bike_allocation)) + ")",
    #              xytext=max_cord, xy=max_cord_shift)

    plt.savefig(
        'ga_800_iter' + str(int(cfg.read_config(['genetic_parameter', 'iteration_time']))) + "_" + str(
            time.strftime('%Y%m%d_%H_%M_%S') + ".pdf"),
        format='pdf')
    plt.show()


if __name__ == '__main__':
    result = []
    for i in range(1):
        ga_instance = config_ga()
        bike_alloc, fitness_value, ga_result = ga_begin(ga_instance)
        result.append((bike_alloc, fitness_value))
        print("=====" + "genetic algorithm round " + str(i + 1) + "=====")
        print("bike allocation is ")
        print(bike_alloc)
        print("fitness value this time:" + str(fitness_value[0]))
        ga_plot(ga_result, bike_alloc)
        # with open("exp_result_" + str(
        #         time.strftime('%Y%m%d_%H_%M_%S') + ".csv"), 'w', encoding='utf-8') as f:
        #     title = 'iteration\\bike_number,'
        #     for station in range(station_count):
        #         title = title + str(station + 1) + ','
        #     title += ', fitness_number,'
        #     f.write(title)
        #     cnt = 1
        #     print(ga_result.generation_best_X)
        #     for best_X in ga_result.generation_best_X:
        #         f.write('\n' + str(cnt) + ",")
        #         cnt += 1
        #         for x_element in best_X:
        #             f.write(str(x_element).replace(".0", "") + ",")
        #         f.write(str(fitness_fun(best_X)).replace(".0", "") + ",")
        # f.close()

    # print(result)
#
# print(rent_6_9)
# ga_instance = config_ga()
# ga_begin(ga_instance)
# fitness_fun([50, 31, 12, 19, 45, 82, 18, 101, 23, 13, 18, 52, 12, 7, 13, 19, 57, 32, 22, 53, 30, 22, 14, 45, 37, 50, 12, 50, 9, 54, 15, 22, 11, 53, 31, 14, 5, 58, 21, 56, 12, 51, 34, 22, 9, 14, 11, 32, 27, 19])