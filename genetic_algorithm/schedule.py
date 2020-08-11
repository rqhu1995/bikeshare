# from random import shuffle
# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# import numpy as np
# import time
# from genetic_algorithm.database import *
# from genetic_algorithm.ge_alg import GeneticAlg
# import matplotlib.pyplot as plt
#
#
# def fitness_fun_model_2(x):
#     """Definition of the fitness function
#     :param x: route of trucks
#     :return:
#     """
#     # print(x)
#     split_point = np.where(x == float(21))[0][0]
#     # print(split_point)
#     routes = [x[0: split_point], x[split_point + 1:]]
#     # user's demand in current hour
#     user_demand = user_800[0]
#     # user's borrow count estimated
#     borrow_amount = np.zeros((station_count, station_count))
#     # bike available in this hour
#     available = initial_bike_count
#     # value of the objective function
#     fval = 0
#     demand = np.array(user_demand)
#     for route in routes:
#         truck_load = 0
#         for truck_no, schedule_station in enumerate(route):
#             load_count = bike_load_calculator(schedule_station - 1, truck_load)
#             truck_load = truck_load - load_count
#             available[schedule_station - 1] = available[schedule_station - 1] + load_count
#     for station_src in range(station_count):
#         for station_dst in range(station_count):
#             if station_src == station_dst:
#                 continue
#             if demand.sum(0)[station_src] <= available[station_src]:
#                 borrow_amount[station_src][station_dst] = user_demand[station_src][station_dst]
#             else:
#                 borrow_amount[station_src][station_dst] = user_demand[station_src][station_dst] // demand.sum(0)[
#                     station_src] * available[station_src]
#             fval += borrow_amount[station_src][station_dst]
#     # print(x)
#     time_cost = time_cost_cal(routes)
#     # print("t:" + str(time_cost))
#     # print("fval:" + str(fval))
#     return phi * time_cost - fval
#
#
# def bike_load_calculator(dest_station, truck_load):
#     """ calculate the amount of bikes up/down load from the truck, + for download, - for upload
#     :param dest_station:
#     :param truck_load:
#     :return:
#     """
#     # print("!!!" + str(dest_station))
#     balance = model_1_target[dest_station] - initial_bike_count[dest_station]
#     if balance >= 0:
#         return min(balance, truck_load)
#     else:
#         return -min(-balance, truck_capacity - truck_load)
#
#
# def time_cost_cal(routes):
#     # print("route" + str(routes))
#     cost = 0
#     # print(routes)
#     for route in routes:
#         current_cost = 0
#         if len(route) < 1:
#             continue
#         current_cost += time_cost_table[0][route[0]]
#         for idx, station_id in enumerate(route[1:]):
#             if idx < len(route) - 1:
#                 current_cost += time_cost_table[route[idx]][route[idx + 1]]
#         current_cost += time_cost_table[route[-1]][station_count]
#         cost += current_cost
#     # print(cost)
#     return cost
#
#
# def generate_new_pop():
#     s = []
#     mylist = [i for i in range(1, station_count + truck_count)]
#     myset = set()
#     chosen_toggle = np.random.choice(10000, population_size, replace=False)
#     while len(myset) < 10000:  # change 5 to however many you want
#         shuffle(mylist)
#         myset.add(tuple(mylist))
#     counter = 0
#     for item in myset:
#         if counter in chosen_toggle:
#             # print(item)
#             s.append(list(item))
#         counter += 1
#     # print("kkk" + str(len(s)))
#     return np.array(s)
#
#
# def customized_mutation(algorithm):
#     mutation_rate = 0.1
#     selected_mutation = 2 * np.random.choice(algorithm.size_pop // 2, int(algorithm.size_pop // 2 * mutation_rate),
#                                              replace=False)
#     for selected in selected_mutation:
#         chosen = algorithm.Chrom[selected].copy()
#         n1, n2 = np.random.randint(0, 21, 2)
#         seg1, seg2 = chosen[n1], chosen[n2]
#         algorithm.Chrom[selected, n1], algorithm.Chrom[selected, n2] = seg2, seg1
#     # print("after" + str(algorithm.Chrom))
#     return algorithm.Chrom
#
#
# def customized_crossover(algorithm):
#     crossover_rate = 0.9
#     Chrom, size_pop, len_chrom = algorithm.Chrom, algorithm.size_pop, algorithm.len_chrom
#     selected_crossover = 2 * np.random.choice(algorithm.size_pop // 2, int(algorithm.size_pop // 2 * crossover_rate),
#                                               replace=False)
#     original_chrom = None
#     for i in selected_crossover:
#         # print(i)
#         original_chrom = [algorithm.Chrom[i].copy(), algorithm.Chrom[i + 1].copy()]
#         # print("before..." + str(original_chrom[0]))
#         # print("before..." + str(original_chrom[1]))
#         spl_1, spl_2 = np.random.choice(19, 2, replace=False)
#         if spl_1 > spl_2:
#             spl_1, spl_2 = spl_2, spl_1
#         # print(spl_1, spl_2)
#         # crossover from the point spl_1 to point spl_2
#         seg = []
#         seg1, seg2 = algorithm.Chrom[i, spl_1: spl_2 + 1].copy(), algorithm.Chrom[i + 1, spl_1: spl_2 + 1].copy()
#         algorithm.Chrom[i, spl_1: spl_2 + 1], algorithm.Chrom[i + 1, spl_1: spl_2 + 1] = seg2, seg1
#
#         for gene_no in range(2):
#             if gene_no == 0:
#                 remaining_genes = [item for item in original_chrom[gene_no] if item not in seg2]
#             else:
#                 remaining_genes = [item for item in original_chrom[gene_no] if item not in seg1]
#             # print(remaining_genes)
#             idx = spl_2 + 1
#             remain_idx = 0
#             while idx != spl_1:
#                 algorithm.Chrom[i + gene_no, idx] = remaining_genes[remain_idx]
#                 idx = (idx + 1) % 21
#                 remain_idx = remain_idx + 1
#         #         print(idx, remain_idx)
#         # print("after..." + str(algorithm.Chrom[i]))
#         # print("after..." + str(algorithm.Chrom[i+1]))
#         return algorithm.Chrom
#
#
# def ga_plot(ga, bike_allocation):
#     # print(ga.generation_best_Y)
#     fig, ax = plt.subplots()
#     generation = []
#     best = []
#     valid_gen = 0
#     for gen in ga.generation_best_Y:
#         if gen < 0:
#             best.append(gen)
#             generation.append(valid_gen)
#             valid_gen = valid_gen + 1
#     plt.xlim((0, valid_gen))
#     plt.ylim((200, 600))
#     plt.xlabel('iteration time')
#     plt.ylabel('fitness value')
#     my_x_ticks = np.arange(0, valid_gen, 5)
#     my_y_ticks = np.arange(200, 600, 100)
#     plt.xticks(my_x_ticks)
#     plt.yticks(my_y_ticks)
#     idx = 0
#     for item in ga.generation_best_X:
#         if np.array_equal(item, bike_allocation):
#             break
#         idx = idx + 1
#     f_val = -fitness_fun_model_2(bike_allocation)
#     max_cord = (idx, int(-fitness_fun_model_2(bike_allocation)))
#     max_cord_shift = (idx + 2, int(-fitness_fun_model_2(bike_allocation)) + 10)
#     ax.plot(np.array(generation),
#             -np.array(best),
#             color='red')
#
#     # plt.vlines(idx, 300, int(f_val), 'black', '--', label='example')
#     # plt.hlines(int(f_val), 0, idx, 'black', '--', label='example')
#     plt.annotate("(" + str(idx) + ", " + str(-fitness_fun_model_2(bike_allocation)) + ")",
#                  xytext=max_cord, xy=max_cord_shift)
#     # script_dir = os.path.dirname(__file__)
#     script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exp_result'))
#     results_dir = os.path.join(script_dir, 'ga_800_schedule_iter/' + time.strftime('%Y%m%d'))
#
#     if not os.path.isdir(results_dir):
#         os.makedirs(results_dir)
#
#     plt.savefig(
#         results_dir + '/ga_800_schedule_iter' + str(time.strftime('%Y%m%d_%H_%M_%S') + ".pdf"),
#         format='pdf')
#     plt.show()
#
#
# if __name__ == '__main__':
#     lb = [1] * (station_count + truck_count - 1)
#     ub = [21] * (station_count + truck_count - 1)
#     ga_instance = GeneticAlg(fitness_fcn=fitness_fun_model_2,
#                              lower_bound=lb, upper_bound=ub,
#                              population_size=population_size,
#                              population_dimension=station_count + truck_count - 1)
#     initial_pop = generate_new_pop()
#     b_X, b_Y, ga_result = ga_instance.run_ga(initial_pop=initial_pop,
#                                              mutation_pub=mutation_prob,
#                                              crossover_method=customized_crossover,
#                                              mutation_method=customized_mutation)
#     for best_X in [b_X]:
#         print("best allocation:" + str(best_X))
#         point = np.where(best_X == 21)[0][0]
#         routes = [best_X[0: point], best_X[point + 1:]]
#         time_cost = time_cost_cal(routes)
#         overall_fitness = -fitness_fun_model_2(best_X)
#         demand_satisfaction = overall_fitness + phi * time_cost
#         print("best route:")
#         for i, route in enumerate(routes):
#             print("truck " + str(i + 1) + ": center -> ", end='')
#             for station in route:
#                 print(str(station) + " -> ", end='')
#             print('center')
#
#         print("demand satisfaction:" + str(demand_satisfaction))
#         print("time_cost:" + str(time_cost))
#         print("overall fitness:" + str(overall_fitness))
#
#     ga_plot(ga_result, b_X)