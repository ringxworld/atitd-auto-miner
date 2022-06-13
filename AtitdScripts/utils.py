import re

import numpy as np
import psutil

from scipy.spatial.distance import cdist

def almost_equal(current, previous):
    equal = True
    assert len(current) == len(previous)
    for idx, val in enumerate(current):
        if abs(previous[idx] - current[idx]) > 10:
            equal = False
    return equal


def extract_match(pattern, search_string):
    p = re.compile(pattern)
    text = p.findall(search_string)
    return text


def find_procs_by_name(name):
    """Return a list of processes matching 'name'."""
    assert name, name
    ls = []
    for p in psutil.process_iter():
        name_, exe, cmdline = "", "", []
        try:
            name_ = p.name()
            cmdline = p.cmdline()
            exe = p.exe()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except psutil.NoSuchProcess:
            continue
        if name == name_ or (len(cmdline) > 0 and cmdline[0] == name) or os.path.basename(exe) == name:
            ls.append(p)
    return ls


def manhattan(A, B):
    return np.abs(A - B).sum()
