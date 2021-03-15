import argparse
from src.load_configs import load_config


def run_pipeline(pipeline_folder):
    config, modules = load_config(pipeline_folder)

    db = modules['db'].get_connection(config)

    if config['loader']['key'] not in db:
        modules['loader'].load_stats(config, db)

    if config['dataset']['key'] not in db:
        modules['dataset'].create_dataset(config, db) # data is current redis db. Here we'll use redis as a cache. 

    if config['trainer']['key'] not in db:
        modules['trainer'].train_model(config, db)

    if config['evaluator']['key'] not in db:
        modules['evaluator'].create_eval_stat(config, db)
        modules['evaluator'].report_eval_stat(config, db)

    if config['visualizer']['key'] not in db:
        modules['visualizer'].create_visuals(config, db)
    modules['visualizer'].display_visuals(config, db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline_folder")
    args = parser.parse_args()
    run_pipeline(args.pipeline_folder)
    