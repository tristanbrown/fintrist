""""""
from fintrist import Study, Recipe

def upgrade():
    full_query = (
        list(Recipe.objects(process='moving_avg')) +
        list(Study.objects(process='moving_avg')))
    for study in full_query:
        parents = study.parents
        try:
            if 'prices' in parents.keys():
                del parents['data']
            else:
                parents['prices'] = parents.pop('data')
            study.update(parents=parents)
        except KeyError:
            pass
        
