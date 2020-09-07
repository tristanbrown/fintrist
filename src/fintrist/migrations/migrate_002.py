""""""
from fintrist import Study

def upgrade():
    for study in Study.objects(process='moving_avg'):
        parents = study.parents
        if 'prices' in parents.keys():
            continue
        try:
            parents['prices'] = parents.pop('data')
        except KeyError:
            pass
        study.update(parents=parents)
