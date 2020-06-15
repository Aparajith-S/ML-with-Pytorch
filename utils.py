import json
from os import path
def getLabelDict(filename):
    if path.exists(filename):
        with open(filename, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None
    return cat_to_name