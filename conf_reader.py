from configparser import ConfigParser

pref = '/Users/hurunqiu/project/static_ffbs/bikeshare/resources/'
cfg_file = pref + "config.ini"

conf = ConfigParser()
conf.read(cfg_file)


def get_val(key, sub_key):
    return conf.get(key, sub_key)
