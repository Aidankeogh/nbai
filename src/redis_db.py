
import redis

class redis_wrapper(redis.Redis):
    def set_namespace(self, namespace):
        self.namespace = namespace
    # overloads getitem/setitem to work like dict
    def __getitem__(self, name):
        return self.get(self.namespace + "/" + name)
    
    def __setitem__(self, name, value):
        self.set(self.namespace + "/" + name, value)

def get_connection(config, host="localhost", port=6379, db=0):
    return redis_wrapper(host=host, port=port, db=db)
