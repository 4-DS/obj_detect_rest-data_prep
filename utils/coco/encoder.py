import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, myobj):
        if isinstance(myobj, np.integer):
            return int(myobj)
        elif isinstance(myobj, np.floating):
            return float(myobj)
        elif isinstance(myobj, np.ndarray):
            return myobj.tolist()
        return json.JSONEncoder.default(self, myobj)


def load_coco_file(coco_file):
    with open(coco_file) as io:
        coco_data = json.load(io)
    return coco_data


def dump_coco_file(file, coco_data):
    with open(file, 'w') as io:
        json.dump(coco_data, io, cls=NumpyEncoder)
