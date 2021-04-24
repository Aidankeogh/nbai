
from os import name
from torch import tensor
import numpy as np
import h5py
import pickle

class h5py_wrapper:
    def __init__(self, name, dir="cache"):
        self.f = h5py.File(dir + '/' + name + '.hdf5', 'w')
        self.set_namespace("root")

    def set_namespace(self, namespace):
        if namespace not in self.f:
            self.f.create_group(namespace)
        self.grp = self.f[namespace]

    # overloads getitem/setitem to work like dict
    def __getitem__(self, key):
        if type(key) is str:
            key = [key]
        key, idx = key[0], key[1:]
        if len(idx) == 0:
            return self.grp[key][:]
        elif type(idx[0]) is str:
            idx = ".".join([str(i) for i in idx])
            return pickle.loads(self.grp[key].attrs[idx])
        else:
            return self.grp[key][idx]

    def __setitem__(self, key, value):
        if type(key) is str:
            key = [key]
        key, idx = key[0], key[1:]

        if key not in self.grp:
            if len(idx) == 0:
                self.grp.create_dataset(key, data=value)
            elif type(idx[0]) is str:
                idx = ".".join([str(i) for i in idx])
                self.grp.create_group(key)
                value = np.void(pickle.dumps(value))
                self.grp[key].attrs[idx] = value
            else:
                raise f"Invalid index, {idx}, cannot create dataset from slice"
        else:
            if len(idx) == 0:
                self.grp[key][:] = value
            elif type(idx[0]) is str:
                idx = ".".join([str(i) for i in idx])
                value = np.void(pickle.dumps(value))
                self.grp[key].attrs[idx] = value
            else:
                self.grp[key][idx] = value

def get_connection(config, name="nbai_db"):
    return h5py_wrapper(name)

if __name__ == "__main__":
    db = get_connection({})
    db['hello'] = [[5, 2, 3, 4, 5], [2, 3, 4, 1, 2]]
    db['hello', 'meta', 0] = {"a": "b"}
    db['hello', 'meta', 1] = {"a": "c"}
    print(db['hello', 'meta', 0])
    print(db['hello', 'meta', 1])
    print(db['hello', :, [3, 4]])