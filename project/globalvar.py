import time

_global_dict = {}
_global_dict_desc = {}


def gl_init():
    global _global_dict
    global _global_dict_desc
    _global_dict = {}
    _global_dict_desc = {}

    _global_dict["start_time"] = int(time.time())

def set_value(name, value, desc=None):
    _global_dict[name] = value
    if desc:
        _global_dict_desc[name] = desc


def get_value(name, def_value=None):
    try:
        return _global_dict[name]
    except KeyError:
        return def_value


def get_desc(name):
    try:
        return _global_dict_desc[name]
    except KeyError:
        return ""
