import numpy as np
import conf_reader as conf
import database as db
# print(conf.cfg_file)
station_count = int(conf.get_val('model_param', 'station_count'))
bike_count = int(conf.get_val('model_param', 'bike_count'))


def fitness_fun_demand(x):
    """Definition of the fitness function
    :param x: the allocation for bicycles at the beginning of each hour
    :return:
    """
    available = (x * 20)
    obj = user_satisfaction(available)
    return -obj


def user_satisfaction(available):
    rent_amount = np.zeros((3, station_count))
    return_amount = np.zeros((3, station_count))
    fval_rent_success = 0
    fval_return_fail = 0
    rent_demand = np.array(db.rent_6_9[:, range(station_count)])
    return_demand = np.array(db.return_6_9[:, range(station_count)])
    for current_stage in range(0, 3):
        for station_id in range(station_count):
            rent_amount[current_stage][station_id] = min(rent_demand[current_stage][station_id], available[station_id])
            return_amount[current_stage][station_id] = min(return_demand[current_stage][station_id],
                                                           max(0,
                                                               db.max_capacity[station_id] - available[station_id]))
            available[station_id] = available[station_id] - rent_amount[current_stage][station_id] + \
                                    return_amount[current_stage][station_id]
            fval_rent_success += rent_amount[current_stage][station_id]
            fval_return_fail += max(0,
                                    return_demand[current_stage][station_id] - return_amount[current_stage][station_id])
    obj = fval_rent_success - fval_return_fail
    return obj
