import re


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
    count = None

    if text:
        count = [int(s) for s in text[0].split() if s.isdigit()]

    if not count:
        return False, 0

    return True, count[0]
