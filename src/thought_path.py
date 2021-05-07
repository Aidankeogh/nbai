import oyaml
import torch
from src.utilities.embedding_utilities import get_name, get_idx

def parse_yaml(yaml_path):
    path = oyaml.load(open(yaml_path, "r"), Loader=oyaml.Loader)

    data_keys = []
    is_embedding = {}
    for k, v in path.items():
        if v['type'] == 'embedding_list':
            is_embedding[k] = True
            for i in range(v['len']):
                data_keys.append(k + "." + str(i))
                is_embedding[k + "." + str(i)] = True
        else:
            data_keys.append(k)
            is_embedding[k] = (v['type'] == 'embedding')

    data_indices = {k: i for i, k in enumerate(data_keys)}

    slice_keys = {}
    for k, v in path.items():
        if v['type'] == 'embedding_list':
            start = data_indices[k + ".0"]
            end = data_indices[k + "." + str(v['len'] - 1)]
            slice_keys[k] = slice(start, end+1)

    choice_keys = {}
    choice_indices = {}
    for k, v in path.items():
        if v['type'] == 'choice':
            if type(v['choices']) is list:
                choices = v['choices']
            else:
                choice_range = list(range(
                    slice_keys[v['choices']].start,
                    slice_keys[v['choices']].stop + 1
                ))
                choices = [data_keys[i] for i in choice_range]
                
            choice_keys[k] = choices
            choice_indices[k] = {choice: i for i, choice in enumerate(choices)}

    return data_keys, data_indices, is_embedding, slice_keys, choice_keys, choice_indices

builtins = {
    "data", 
    "data_keys", 
    "data_indices", 
    "is_embedding",
    "slice_keys",
    "choice_keys",
    "choice_indices",
    "data_key_set",
}
class thought_object():
    def __init__(
        self, 
        data_keys,
        data_indices,
        is_embedding,
        slice_keys,
        choice_keys,
        choice_indices,
        data=None):

        self.data_keys = data_keys
        self.data_indices = data_indices
        self.is_embedding = is_embedding
        self.slice_keys = slice_keys
        self.choice_keys = choice_keys
        self.choice_indices = choice_indices
        self.data_key_set = set(data_keys + list(slice_keys.keys()))

        if data is None:
            self.data = torch.zeros(len(data_keys))
        else:
            self.data = data

    def is_data(self, key):
        return key in self.data_key_set

    def keys(self):
        return self.data_keys

    def __getattr__(self, key):
        if key in builtins:
            return super().__getattr__(key)
        elif self.is_data(key):
            if key in self.slice_keys:
                out = self.data[self.slice_keys[key]]
                if self.is_embedding[key]:
                    out = [get_name(int(o.item())) for o in out]
            else:
                out = self.data[self.data_indices[key]]
                if self.is_embedding[key]:
                    out = get_name(int(out.item()))
                elif key in self.choice_keys:
                    out = self.choice_keys[key][int(out.item())]
            return out
        else:
            raise Exception(f"ERROR, attribute {key} not found")

    def __setattr__(self, key, value):
        if key in builtins: # all keys that we need to operate on regularly
            return super().__setattr__(key, value)
        elif self.is_data(key):
            if key in self.slice_keys:
                if self.is_embedding[key]:
                    value = [get_idx(o) for o in value]
                self.data[self.slice_keys[key]] = torch.tensor(value)
            else:
                if self.is_embedding[key]:
                    value = get_idx(value)
                elif key in self.choice_indices:
                    value = self.choice_indices[key][value]
                self.data[self.data_indices[key]] = value
        else:
            raise Exception(f"ERROR, attribute {key} not found")


if __name__ == "__main__":
    args = parse_yaml("src/data/play.yaml")

    t = thought_object(*args)
