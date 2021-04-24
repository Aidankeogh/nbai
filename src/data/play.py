from src.embedding_utilities import get_name, get_idx
from src.thought_path import parse_yaml
import torch
(
    data_keys,
    data_indices,
    is_embedding,
    slice_keys,
    choice_keys,
    choice_indices
) = parse_yaml("src/data/play.yaml")

data_key_set = set(data_keys + list(slice_keys.keys()))
def is_data(key):
    return key in data_key_set

class Play():
    def __init__(self, data=None, data_id=None):
        if data is None:
            self.data = torch.zeros(len(data_keys))
        else:
            self.data = data

    def keys(self):
        return data_keys

    def __getattr__(self, key):
        if is_data(key):
            if key in slice_keys:
                out = self.data[slice_keys[key]]
                if is_embedding[key]:
                    out = [get_name(int(o.item())) for o in out]
            else:
                out = self.data[data_indices[key]]
                if is_embedding[key]:
                    out = get_name(int(out.item()))
                elif key in choice_keys:
                    out = choice_keys[key][int(out.item())]
            return out
        elif key == "score_change":
            return (self.shot_made * (2 + self.is_3) + 
                    self.first_free_throw_made + 
                    self.middle_free_throw_made + 
                    self.last_free_throw_made)
        else:
            raise Exception(f"ERROR, attribute {key} not found")

    def __setattr__(self, key, value):
        if is_data(key):
            if key in slice_keys:
                if is_embedding[key]:
                    value = [get_idx(o) for o in value]
                self.data[slice_keys[key]] = torch.tensor(value)
            else:
                if is_embedding[key]:
                    value = get_idx(value)
                elif key in choice_indices:
                    value = choice_indices[key][value]
                self.data[data_indices[key]] = value
        elif key == "data":
            return super().__setattr__(key, value)
        else:
            raise Exception(f"ERROR, attribute {key} not found")

if __name__ == "__main__":
    p = Play()
    curry = "stephen-curry"
    lebron = "lebron-james"
    kd = "kevin-durant"
    p.initial_event = "shot"
    p.shooter = curry
    p.offense_roster = [curry, curry, kd, kd, kd]
    p.defense_roster = [lebron, lebron, lebron, lebron, lebron]
    assert(p.offense_roster[1] == "stephen-curry")
    assert(p.shooter == "stephen-curry")
    print(p.offense_roster)

