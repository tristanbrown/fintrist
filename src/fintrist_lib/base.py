"""Base classes for analytical processes."""

class RecipeBase():

    studyname = None
    parents = None
    parent_params = None
    valid_type = None

    def __repr__(self):
        return self.studyname

    @staticmethod
    def process(*args, **kwargs):
        pass
