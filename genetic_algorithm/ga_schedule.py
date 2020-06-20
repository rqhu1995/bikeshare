import time

import matplotlib.pyplot as plt
import numpy as np
from sko.GA import GA
from sko.operators import crossover

import config.config_reader as cfg
from data import hub, user_demand as ud

# station_count
station_count = int(cfg.read_config('station_count'))

# truck count
truck_count = int(cfg.read_config('truck_count'))


def fitness_fun(x):
    """Definition of the fitness function
    :param x: route of trucks
    :return:
    """
    # user's demand in current hour
    user_demand = ud.user_800[0]
    # user's borrow count estimated
    borrow_amount = np.zeros((station_count, station_count))
    # bike available in this hour
    available = x
    # value of the objective function
    fval = 0
    demand = np.array(user_demand)
    for station_src in range(station_count):
        for station_dst in range(station_count):
            if station_src == station_dst:
                continue
            if demand.sum(0)[station_src] <= available[station_src]:
                borrow_amount[station_src][station_dst] = user_demand[station_src][station_dst]
            else:
                borrow_amount[station_src][station_dst] = user_demand[station_src][station_dst] // demand.sum(0)[
                    station_src] * available[station_src]
            fval += borrow_amount[station_src][station_dst]
    return -fval


def config_ga():
    constraint_eq = [
        lambda x: int(cfg.read_config('bike_total_count')) - sum(x)
    ]
    lb = [0] * station_count
    ub = hub.hub
    iteration = int(cfg.read_config("iteration_time"))
    ga = GA(func=fitness_fun, size_pop=300, n_dim=station_count, max_iter=iteration, prob_mut=0.01, lb=lb, ub=ub,
            constraint_eq=constraint_eq,
            precision=[1] * 20)
    if cfg.read_config("crossover_type") is not None:
        ga.register(operator_name='crossover', operator=crossover.crossover_1point)
    # TODO: 变异和排序的策略配置（暂时不考虑）
    return ga


def ga_begin(ga):
    best_x, best_y = ga.run()
    return best_x, best_y, ga


def ga_plot(ga, bike_allocation):
    # print(ga.generation_best_Y)
    fig, ax = plt.subplots()
    generation = []
    best = []
    valid_gen = 0
    for gen in ga.generation_best_Y:
        if gen < 0:
            best.append(gen)
            generation.append(valid_gen)
            valid_gen = valid_gen + 1
    plt.xlim((0, valid_gen))
    plt.ylim((300, 510))
    plt.xlabel('iteration time')
    plt.ylabel('fitness value')
    my_x_ticks = np.arange(0, valid_gen, 5)
    my_y_ticks = np.arange(300, 510, 50)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    idx = 0
    for item in ga.generation_best_X:
        if np.array_equal(item, bike_allocation):
            break
        idx = idx + 1
    f_val = -fitness_fun(bike_allocation)
    max_cord = (idx, int(-fitness_fun(bike_allocation)))
    max_cord_shift = (idx + 2, int(-fitness_fun(bike_allocation)) + 10)
    ax.plot(np.array(generation),
            -np.array(best),
            color='red')

    plt.vlines(idx, 300, int(f_val), 'black', '--', label='example')
    plt.hlines(int(f_val), 0, idx, 'black', '--', label='example')
    plt.annotate("(" + str(idx) + ", " + str(-fitness_fun(bike_allocation)) + ")",
                 xytext=max_cord, xy=max_cord_shift)

    plt.savefig(
        'ga_800_iter' + str(int(cfg.read_config('iteration_time'))) + "_" + str(
            time.strftime('%Y%m%d_%H_%M_%S') + ".pdf"),
        format='pdf')
    plt.show()


if __name__ == '__main__':
    result = []
    for i in range(20):
        ga_instance = config_ga()
        bike_alloc, fitness_value, ga_result = ga_begin(ga_instance)
        result.append((bike_alloc, fitness_value))
        print("=====" + "genetic algorithm round " + str(i + 1) + "=====")
        print("bike allocation is ")
        print(bike_alloc)
        print("fitness value this time:" + str(fitness_value[0]))
        ga_plot(ga_result, bike_alloc)
        with open("exp_result_" + str(
                time.strftime('%Y%m%d_%H_%M_%S') + ".csv"), 'w', encoding='utf-8') as f:
            title = 'iteration\\bike_number,'
            for station in range(station_count):
                title = title + str(station + 1) + ','
            title += ', fitness_number,'
            f.write(title)
            cnt = 1
            for best_X in ga_result.generation_best_X:
                f.write('\n' + str(cnt + 1))
                cnt += 1
                for x_element in best_X:
                    f.write(str(x_element).replace(".0", "") + ",")
                f.write(str(fitness_fun(best_X)).replace(".0", "") + ",")
        f.close()

    print(result)
