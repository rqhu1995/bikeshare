import configparser

cfg_file = "E:\\bikeshare\\resources\\config.ini"

conf = configparser.ConfigParser()
conf.read(cfg_file)


def get_val(key, sub_key):
    return conf.get(key, sub_key)
