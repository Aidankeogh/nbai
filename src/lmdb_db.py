
import lmdb

class lmdb_wrapper:
    def __init__(self, name, map_size):
        self.env = lmdb.open(name, map_size=map_size)
        self.namespace = ""

    def set_namespace(self, namespace):
        self.namespace = namespace

    # overloads getitem/setitem to work like dict
    def __getitem__(self, name):
        with self.env.begin() as txn:
            key = (self.namespace + "/" + name).encode()
            return txn.get(key)
    
    def __setitem__(self, name, value):
        with self.env.begin(write=True) as txn:
            key = (self.namespace + "/" + name).encode()
            txn.put(key, value)

def get_connection(config, name="nbai_lmdb", map_size=30 * 1e9): # 30 gb database size
    return lmdb_wrapper(name, map_size)
