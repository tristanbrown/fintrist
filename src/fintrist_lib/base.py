"""Base classes for analytical processes."""
import inspect as ins

class RecipeBase():

    studyname = None
    parents = {}
    params = {}
    parent_params = {}
    valid_type = None

    def __repr__(self):
        return self.studyname

    @property
    def __name__(self):
        return self.__class__.__name__

    @staticmethod
    def process(*args, **kwargs):
        pass

    @classmethod
    def get_params(cls):
        sig = ins.signature(cls.process)
        args = sig.parameters.keys()
        return [arg for arg in args if arg not in cls.parents.keys()]

    @classmethod
    def see_args(cls):
        print(f"Parents: {list(cls.parents.keys())}")
        print(f"Params: {cls.get_params()}")
