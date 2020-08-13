# -*- coding:utf-8 _*-
""" 
@author:Runqiu Hu
@license: Apache Licence 
@file: sa.py 
@time: 2020/08/13
@contact: hurunqiu@live.com
@project: bikeshare rebalancing

* Cooperating with Dr. Matt in 2020
"""
import conf_reader as conf
from database import cluster_info
from sko.SA import SA_TSP
from static_bikeshare.rebalancing_route import fitness_function_rebalancing, generate_feasible_population

station_count = int(conf.get_val('model_param', 'station_count'))
truck_count = int(conf.get_val('model_param', 'truck_count'))


def run_SA():
    all_stations = [station['station_id'] for station in cluster_info[3]]
    x0 = generate_feasible_population(all_stations, truck_count, 50)[0]
    # print(x0)
    sa_tsp = SA_TSP(func=fitness_function_rebalancing, x0=x0, T_max=1000, T_min=1, L=1000)
    best_points, best_distance = sa_tsp.run()
    print(best_points, best_distance)
    # min_val = np.inf
    # best_routes = None
    # for k in sa_gene_record[str(best_points)]:
    #     # print(fitness_function(k)[0])
    #     first['inserted'] = True
    #     if fitness_function(k)[0] < min_val:
    #         min_val = fitness_function(k)[0]
    #         best_routes = k
    # print(min_val, best_routes)
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(sa_tsp.best_y_history)
    # ax[0].set_xlabel("Iteration")
    # ax[0].set_ylabel("Distance")
    # final_routes = []
    # for item in best_routes:
    #     if item < 0:
    #         final_routes.append([])
    #         continue
    #     r = final_routes[-1]
    #     r.append(item)
    #
    # # print(location_lat_lon.loc)
    # for pic in range(1, 3):
    #     lon_list = []
    #     lat_list = []
    #     for station in final_routes[pic - 1]:
    #         lon_list.append(schedule_station_info[station]['X'])
    #         lat_list.append(schedule_station_info[station]['Y'])
    #     ax[pic].plot(lon_list, lat_list, marker='o', markerfacecolor='b', color='c', linestyle='-')
    #     ax[pic].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #     ax[pic].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #     ax[pic].set_xlabel("Longitude")
    #     ax[pic].set_ylabel("Latitude")
    # plt.savefig('schedule.png')
    # plt.show()


run_SA()
