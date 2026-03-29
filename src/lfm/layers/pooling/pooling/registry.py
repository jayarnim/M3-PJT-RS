POOLING_REGISTRY = {}

def register(name):
    def wrapper(cls):
        POOLING_REGISTRY[name] = cls
        return cls
    return wrapper