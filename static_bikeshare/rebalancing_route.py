import json

import numpy as np
from geopy.distance import geodesic
from more_itertools import pairwise
from numpyencoder import NumpyEncoder
import conf_reader as conf
from database import cluster_info, station_info
from database import max_capacity, initial_bike, distance_matrix, cluster_center
from static_bikeshare.demand_calculation import user_satisfaction

truck_count = int(conf.get_val('model_param', 'truck_count'))
truck_capacity = int(conf.get_val('model_param', 'truck_capacity'))
truck_velocity = int(conf.get_val('model_param', 'truck_velocity'))
time_cost = int(conf.get_val('model_param', 'time_cost'))
selected_cluster = int(conf.get_val('model_param', 'selected_cluster'))
station_list = [station['station_id'] for station in cluster_info[selected_cluster]]
crossover_rate = float(conf.get_val('ga_param', 'crossover_rate'))
mutation_rate = float(conf.get_val('ga_param', 'mutation_rate'))
pop_size = int(conf.get_val('ga_param', 'population_size'))
station_count = int(conf.get_val('model_param', 'station_count'))


def truck_route_traverse(routes):
    # print(routes)
    allocation = initial_bike
    route_result = []
    for route in routes:
        route_info = {
            'route': route,
            'distance_time': 0,
            'working_time': 0,
            'actual_allocation': []
        }
        truck_inventory = 0
        for station, next_station in pairwise(route):
            station_demand = station_info[station]['demand']
            if station_demand > 0:
                delivery = min(truck_inventory, station_demand)
                # print(station_demand)
                truck_inventory -= delivery
                allocation[station] = min(allocation[station] + delivery, max_capacity[station])
                route_info['actual_allocation'].append(delivery)
            elif station_demand < 0:
                pickup = min(-station_demand, truck_capacity - truck_inventory)
                truck_inventory += pickup
                allocation[station] = max(allocation[station] - pickup, 0)
                route_info['actual_allocation'].append(-pickup)
            route_info['distance_time'] += distance_matrix[station][next_station] / truck_velocity
            route_info['working_time'] += 1
        # print(cluster_center[selected_cluster])
        dist_center_to_first = geodesic(
            (station_info[route[0]]['latitude'], station_info[route[0]]['longitude']),
            cluster_center[selected_cluster]).m
        # print(dist_center_to_first)
        route_info['distance_time'] += 2 * dist_center_to_first / truck_velocity

        route_result.append(route_info)
    with open('/Users/hurunqiu/project/static_ffbs/bikeshare/route_info.json', 'a+', encoding='utf-8') as fp_route_info:
        json.dump(route_result, fp_route_info, cls=NumpyEncoder)
    return allocation, route_result


def fitness_function_rebalancing(chrom):
    # print(chrom)
    """
    一条路线确定好后，完整的调度成本计算
    :param routes -- route数组的list
    :return:
    """
    routes = chromosome2routelist(chrom)
    initial_available, all_routes_result = truck_route_traverse(routes)
    with open('E:\\bikeshare\\result.json', 'w+', encoding='utf-8') as route_result:
        json.dump(str(all_routes_result), fp=route_result)
    fval_user = user_satisfaction(initial_available)
    max_time = -1
    for route in all_routes_result:
        route_time = route['working_time'] + route['distance_time']
        if route_time >= max_time:
            max_time = route_time
    fval_cart = max_time * time_cost
    return fval_cart - fval_user


def station_permutation(all_stations, population_size):
    station_to_np = np.array(all_stations)
    complete = []
    perms = set()
    # (1) Draw N samples from permutations Universe U (#U = k!)
    for i in range(population_size):
        # (2) Endless loop
        while True:
            # (3) Generate a random permutation form U
            perm = np.random.permutation(len(station_to_np))
            key = tuple(perm)
            # (4) Check if permutation already has been drawn (hash table)
            if key not in perms or station_to_np[perm] < 0:
                # (5) Insert into set
                perms.update(key)
                # (6) Break the endless loop
                break
        complete.append(station_to_np[perm])
    return complete


# 生成初始可行解（输入全部站点，卡车数量）
def generate_feasible_population(all_stations, truck_count, population_size):
    """
    :param all_stations: 全部调度站点编号
    :param truck_count: 卡车数量
    :param population_size: 种群个数
    :return: 在站点编号中插入(卡车数量-1)个负数的population_size个染色体
    """
    # 生成组合
    all_station_perm = station_permutation(all_stations, population_size)
    inserted_genes = []
    # 插空
    for permute in all_station_perm:
        counter = 0
        while counter < truck_count - 1:
            permute = np.insert(permute, (counter + 1) * len(permute + counter) // truck_count, -counter)
            counter += 1
        inserted_genes.append(np.array(permute).reshape((-1,)))
    # print(inserted_genes)
    return np.array(inserted_genes).reshape((population_size, -1))


def ordered_cross_over(chrom_1, chrom_2):
    gene_num = station_count
    new_chrome_1 = np.full((gene_num,), -100)
    new_chrome_2 = np.full((gene_num,), -100)
    inserted = [[], []]
    inserted[0] = [False for i in range(gene_num)]
    inserted[1] = [False for i in range(gene_num)]
    rand_num = np.random.randint(0, 2)
    if rand_num == 1:
        split_points = [gene_num // 3 - 1, 2 * (gene_num // 3)]
    else:
        split_points = [gene_num // 3, 2 * (gene_num // 3) + 1]

    new_chrome_1[split_points[0] + 1: split_points[1] + 1] = chrom_1[split_points[0] + 1: split_points[1] + 1]
    new_chrome_2[split_points[0] + 1: split_points[1] + 1] = chrom_2[split_points[0] + 1: split_points[1] + 1]

    # print(new_chrome_1)
    inserted[0][split_points[0] + 1: split_points[1] + 1] = [True] * (split_points[1] - split_points[0])
    inserted[1][split_points[0] + 1: split_points[1] + 1] = [True] * (split_points[1] - split_points[0])
    origin_idx = split_points[1] + 1
    new_idx = split_points[1] + 1
    while False in inserted[1]:
        while chrom_1[origin_idx] in new_chrome_2:
            origin_idx = (origin_idx + 1) % gene_num
        new_chrome_2[new_idx] = chrom_1[origin_idx]
        inserted[1][new_idx] = True
        new_idx = (new_idx + 1) % gene_num
        # print(new_chrome_2)
    origin_idx = split_points[1] + 1
    new_idx = split_points[1] + 1
    while False in inserted[0]:
        while chrom_2[origin_idx] in new_chrome_1:
            origin_idx = (origin_idx + 1) % gene_num
        new_chrome_1[new_idx] = chrom_2[origin_idx]
        inserted[0][new_idx] = True
        new_idx = (new_idx + 1) % gene_num
    return new_chrome_1, new_chrome_2


# 可行解的交叉：非负位置
def customized_crossover(algorithm):
    random_choice = np.random.choice(pop_size // 2, int(np.floor(pop_size // 2 * 0.8)), replace=False)

    for selected_chrom_id in random_choice:
        chrom_cpy_1 = algorithm.Chrom[selected_chrom_id][algorithm.Chrom[selected_chrom_id] > 0].copy()
        chrom_cpy_2 = algorithm.Chrom[selected_chrom_id * 2][algorithm.Chrom[selected_chrom_id * 2] > 0].copy()
        new_chrome_1, new_chrome_2 = ordered_cross_over(chrom_cpy_1, chrom_cpy_2)
        counter = 0
        while counter < truck_count - 1:
            new_chrome_1 = np.insert(new_chrome_1, (counter + 1) * (len(new_chrome_1) + counter) // truck_count,
                                     -counter)
            new_chrome_2 = np.insert(new_chrome_2, (counter + 1) * (len(new_chrome_2) + counter) // truck_count,
                                     -counter)
            counter += 1
        algorithm.Chrom[selected_chrom_id], \
        algorithm.Chrom[selected_chrom_id * 2] = \
            new_chrome_1, new_chrome_2
    # print("~~~~~")
    # print(new_chrome_1, new_chrome_2)
    return algorithm.Chrom


# 可行解的变异
def customized_mutation(algorithm):
    selected_mutation = 2 * np.random.choice(station_count // 2, int(station_count // 2 * mutation_rate),
                                             replace=False)
    for selected in selected_mutation:
        chosen = algorithm.Chrom[selected].copy()
        chosen = chosen[chosen > 0]
        n1, n2 = np.random.randint(0, station_count, 2)
        seg1, seg2 = chosen[n1], chosen[n2]
        chosen[n1], chosen[n2] = seg2, seg1
        counter = 0
        while counter < truck_count - 1:
            chosen = np.insert(chosen, (counter + 1) * (len(chosen) + counter) // truck_count, -counter)
            counter += 1
        algorithm.Chrom[selected] = chosen
    return algorithm.Chrom


# 将染色体转换为route的list
def chromosome2routelist(chrom):
    routes = []
    route = []
    for gene in chrom:
        if gene > 0:
            route.append(gene)
        else:
            routes.append(route)
            route = []
    routes.append(route)
    return routes
