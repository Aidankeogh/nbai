import yaml
import src.pipeline_utilities.load_configs
path = yaml.safe_load(open("src/play_path.yaml", "r"))
print(path)