def get_connection(config):
    return {}

def load_stats(config, db):
    db[config['loader']['key']] = [1, 2, 3]

def create_dataset(config, db):
    input_stats = db[config['loader']['key']]
    dataset = tuple(input_stats)
    db[config['dataset']['key']] = dataset

def train_model(config, db):
    dataset = db[config['dataset']['key']]
    params = config['trainer']['params']
    p =  params['p']

    avg = sum(dataset) / len(dataset)
    db[config['trainer']['key']] = lambda a : a - p / avg

def create_eval_stat(config, db): 
    input_stats = db[config['loader']['key']]
    model = db[config['trainer']['key']]
    params = config['evaluator']['params']
    k = params['k']

    eval_stat = [model(x) * k for x in input_stats]
    db[config['evaluator']['key']] = eval_stat

def report_eval_stat(config, db):
    eval_stat = db[config['evaluator']['key']]
    print(config['evaluator']['key'], eval_stat)

def create_visuals(config, db):
    stat = db[config['evaluator']['key']]
    db[config['visualizer']['key']] = "---____" + str(stat) + "____---"

def display_visuals(config, db):
    visual = db[config['visualizer']['key']]
    print(visual)
