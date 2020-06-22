import time
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sko.GA import GA
from sko.operators import crossover, mutation

import config.config_reader as cfg
from data import hub, user_demand as ud, desired_target as tgt

# multi-objective parameter \phi
phi = int(cfg.read_config('phi'))

# station_count
station_count = int(cfg.read_config('station_count'))

# truck count
truck_count = int(cfg.read_config('truck_count'))

# desired schedule target
desired_schedule_target = tgt.target

# initial bike count
initial_bike = ud.initial_bike_count

# truck_capacity
truck_capacity = int(cfg.read_config('truck_capacity'))


def fitness_fun(x):
    """Definition of the fitness function
    :param x: route of trucks
    :return:
    """
    # print("!!!!!!")
    # print(x)
    split_point = np.where(x == float(21))[0][0]
    # print(split_point)
    routes = [x[0: split_point], x[split_point + 1:]]
    # user's demand in current hour
    user_demand = ud.user_800[0]
    # user's borrow count estimated
    borrow_amount = np.zeros((station_count, station_count))
    # bike available in this hour
    available = initial_bike
    # value of the objective function
    fval = 0
    demand = np.array(user_demand)
    truck_load = 0
    for route in routes:
        for truck_no, schedule_station in enumerate(route):
            if schedule_station == 21:
                continue
            load_count = bike_load_calculator(schedule_station - 1, truck_load)
            truck_load = truck_load - load_count
            available[schedule_station - 1] = available[schedule_station - 1] + load_count
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
    time_cost = time_cost_cal(routes)
    print(time_cost)
    print(fval)
    return -(phi * time_cost - fval)


def generate_new_pop():
    s = []
    mylist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    myset = set()
    chosen_toggle = np.random.choice(10000, 100, replace=False)
    while len(myset) < 10000:  # change 5 to however many you want
        shuffle(mylist)
        myset.add(tuple(mylist))
    counter = 0
    for item in myset:
        if counter in chosen_toggle:
            s.append(list(item))
        counter += 1
    # print("kkk" + str(len(s)))
    return np.array(s)


def bike_load_calculator(dest_station, truck_load):
    """ calculate the amount of bikes up/down load from the truck, + for download, - for upload
    :param dest_station:
    :param truck_load:
    :return:
    """
    # print("!!!" + str(dest_station))
    balance = desired_schedule_target[dest_station] - initial_bike[dest_station]
    if balance >= 0:
        return min(balance, truck_load)
    else:
        return -min(-balance, truck_capacity - truck_load)


def time_cost_cal(routes):
    # print("route" + str(routes))
    cost = 0
    for route in routes:
        if len(route <= 2):
            continue
        for idx, station_id in enumerate(route):
            if idx < len(route) - 1:
                cost += ud.time_cost_table[route[idx]][route[idx + 1]]
    return cost


def customized_mutation(algorithm):
    mutation_rate = 0.1
    selected_mutation = 2 * np.random.choice(algorithm.size_pop // 2, int(algorithm.size_pop // 2 * mutation_rate),
                                             replace=False)
    for selected in selected_mutation:
        chosen = algorithm.Chrom[selected].copy()
        n1, n2 = np.random.randint(0, 21, 2)
        seg1, seg2 = chosen[n1], chosen[n2]
        algorithm.Chrom[selected, n1], algorithm.Chrom[selected, n2] = seg2, seg1
    # print("after" + str(algorithm.Chrom))
    return algorithm.Chrom


def customized_crossover(algorithm):
    crossover_rate = 0.9
    Chrom, size_pop, len_chrom = algorithm.Chrom, algorithm.size_pop, algorithm.len_chrom
    selected_crossover = 2 * np.random.choice(algorithm.size_pop // 2, int(algorithm.size_pop // 2 * crossover_rate),
                                              replace=False)
    original_chrom = None
    for i in selected_crossover:
        # print(i)
        original_chrom = [algorithm.Chrom[i].copy(), algorithm.Chrom[i + 1].copy()]
        # print("before..." + str(original_chrom[0]))
        # print("before..." + str(original_chrom[1]))
        spl_1, spl_2 = np.random.choice(19, 2, replace=False)
        if spl_1 > spl_2:
            spl_1, spl_2 = spl_2, spl_1
        # print(spl_1, spl_2)
        # crossover from the point spl_1 to point spl_2
        seg = []
        seg1, seg2 = algorithm.Chrom[i, spl_1: spl_2 + 1].copy(), algorithm.Chrom[i + 1, spl_1: spl_2 + 1].copy()
        algorithm.Chrom[i, spl_1: spl_2 + 1], algorithm.Chrom[i + 1, spl_1: spl_2 + 1] = seg2, seg1

        for gene_no in range(2):
            if gene_no == 0:
                remaining_genes = [item for item in original_chrom[gene_no] if item not in seg2]
            else:
                remaining_genes = [item for item in original_chrom[gene_no] if item not in seg1]
            print(remaining_genes)
            idx = spl_2 + 1
            remain_idx = 0
            while idx != spl_1:
                algorithm.Chrom[i + gene_no, idx] = remaining_genes[remain_idx]
                idx = (idx + 1) % 21
                remain_idx = remain_idx + 1
        #         print(idx, remain_idx)
        # print("after..." + str(algorithm.Chrom[i]))
        # print("after..." + str(algorithm.Chrom[i+1]))
        return algorithm.Chrom


def config_ga():
    lb = [1] * (station_count + truck_count - 1)
    ub = [21] * (station_count + truck_count - 1)
    iteration = int(cfg.read_config("iteration_time"))
    ga = GA(func=fitness_fun, size_pop=100, n_dim=station_count + truck_count - 1, max_iter=iteration, prob_mut=0.01,
            lb=lb, ub=ub,
            precision=[1] * (station_count + truck_count - 1))
    ga.Chrom = generate_new_pop()
    if cfg.read_config("crossover_type") is not None:
        ga.register(operator_name='crossover', operator=customized_crossover). \
            register(operator_name='mutation', operator=customized_mutation)
    # TODO: 排序的策略配置（暂时不考虑）
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
    plt.ylim((600, 1000))
    plt.xlabel('iteration time')
    plt.ylabel('fitness value')
    my_x_ticks = np.arange(0, valid_gen, 5)
    my_y_ticks = np.arange(600, 1000, 50)
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
    print("start_running")
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
