import importlib
import yaml
import os

def load_config(package):
    folder_path = os.path.join('pipeline_configs', package)

    config = {}
    modules = {}
    components = ['db', 'loader', 'dataset', 'trainer', 'evaluator', 'visualizer']
    for component in components:
        config[component] = yaml.safe_load(open(os.path.join(folder_path, component + '.yaml'), 'r'))
        config[component]['key'] = "(" + component + "." + config[component]['version'] + ")"
        for dependency in config[component]['deps']:
            config[component]['key'] += '-' + config[dependency]['key']
        modules[component] = importlib.import_module("src." + config[component]['module'])

    return config, modules
