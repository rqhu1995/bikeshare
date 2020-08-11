import yaml
import os


class Config:
    @staticmethod
    def read_config(config_name):
        """
        :param config_name:
        property key:
        [iteration_time, station_count, bike_total_count]
        :return:
        the value of the properties
        """
        file_name_path = os.path.split(os.path.realpath(__file__))[0]
        yaml_path = os.path.join(file_name_path, '../resources/ga_config.yml')
        f = open(yaml_path, 'r', encoding='utf-8')
        content = f.read()
        yaml_dict = yaml.safe_load(content)[config_name[0]]
        value = yaml_dict[config_name[1]]
        if config_name[1] not in yaml_dict.keys():
            raise Exception("Property " + config_name[1] + " doesn't exist")
        # print(value)
        return value
