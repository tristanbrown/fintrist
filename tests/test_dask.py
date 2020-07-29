"""Structures useful for testing dask."""

class Obj():
    def __init__(self, name, delay, inputs=None):
        self.name = name
        self.delay = delay
        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
    
    def wait(self, dummy):
        from time import sleep
        print(f"{self.name}: Start waiting for {self.delay} sec")
        sleep(self.delay)
        print(f"{self.name}: Done waiting for {self.delay} sec")
    
db = {
    'f1': Obj('f1', 10),
    'f2': Obj('f2', 20),
    'f3': Obj('f3', 5, ['f1', 'f2']),
    'f4': Obj('f4', 30),
    'f5': Obj('f5', 1, ['f3', 'f4'])
}
dsk = {key: (val.wait, val.inputs) for key, val in db.items()}
