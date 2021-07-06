import oyaml
import torch
from src.utilities.embedding_utilities import get_name, get_idx

import oyaml


class DataConfig:
    def __init__(self, yaml_path):
        path = oyaml.load(open(yaml_path, "r"), Loader=oyaml.Loader)

        self.data_keys = []
        self.triggers = {}
        self.types = {}
        self.is_embedding = {}
        self.is_int = {}
        self.is_choice = {}
        self.slice_keys = {}
        self.data_indices = {}
        self.choice_keys = {}
        self.choice_indices = {}
        self.embedding_choices = {}
        for k, v in path.items():
            if v["type"] == "embedding_list":
                self.is_embedding[k] = True
                self.is_int[k] = False
                self.is_choice[k] = False
                for i in range(v["len"]):
                    sub_key = k + "_" + str(i)
                    self.data_indices[sub_key] = len(self.data_keys)
                    self.data_keys.append(sub_key)
                    self.is_embedding[sub_key] = True
                    self.is_int[sub_key] = False
                start = self.data_indices[k + "_0"]
                end = len(self.data_keys)
                self.slice_keys[k] = slice(start, end)
            elif v["type"] == "choice":
                self.is_embedding[k] = False
                self.is_int[k] = False
                self.is_choice[k] = True
                if type(v["choices"]) is list:
                    choices = v["choices"]
                else:
                    choice_range = list(
                        range(
                            self.slice_keys[v["choices"]].start,
                            self.slice_keys[v["choices"]].stop + 1,
                        )
                    )
                    choices = [self.data_keys[i] for i in choice_range]
                for choice in choices:
                    sub_key = k + "_" + str(choice).lower()
                    self.data_indices[sub_key] = len(self.data_keys)
                    self.data_keys.append(sub_key)
                    self.is_embedding[sub_key] = False
                    self.is_int[sub_key] = False
                start = self.data_indices[k + "_" + str(choices[0]).lower()]
                end = len(self.data_keys)
                self.slice_keys[k] = slice(start, end)
                self.choice_keys[k] = choices
                self.choice_indices[k] = {choice: i for i, choice in enumerate(choices)}
            else:
                self.data_indices[k] = len(self.data_keys)
                self.data_keys.append(k)
                self.is_embedding[k] = v["type"] == "embedding"
                if "choices" in v and self.is_embedding[k]:
                    self.embedding_choices[k] = v["choices"]
                self.is_int[k] = v["type"] == "int"

            if "triggers" in v:
                self.triggers[k] = v["triggers"]
            self.types[k] = v["type"]

            self.data_key_set = set(self.data_keys + list(self.slice_keys.keys()))


builtins = {
    "data",
    "data_keys",
    "data_indices",
    "is_embedding",
    "is_int",
    "is_choice",
    "slice_keys",
    "choice_keys",
    "choice_indices",
    "data_key_set",
    "triggers",
    "types",
}


class ThoughtPath:
    def __init__(self, data_config, data=None):

        self.data_keys = data_config.data_keys
        self.data_indices = data_config.data_indices
        self.is_embedding = data_config.is_embedding
        self.is_int = data_config.is_int
        self.slice_keys = data_config.slice_keys
        self.choice_keys = data_config.choice_keys
        self.choice_indices = data_config.choice_indices
        self.triggers = data_config.triggers
        self.types = data_config.types
        self.data_key_set = data_config.data_key_set
        self.is_choice = data_config.is_choice

        if data is None:
            self.data = torch.zeros(len(self.data_keys))
        else:
            self.data = torch.Tensor(data)

    def is_data(self, key):
        return key in self.data_key_set

    def keys(self):
        return self.data_keys + list(self.slice_keys.keys())

    def __getattr__(self, key):
        if key in builtins and hasattr(self, key):
            return super().__getattr__(key)
        elif self.is_data(key):
            if key in self.slice_keys:
                out = self.data[self.slice_keys[key]]
                if self.is_embedding[key]:
                    out = [get_name(int(o.item())) for o in out]
                if self.is_int[key]:
                    out = [int(out.item()) for o in out]
                if self.is_choice[key]:
                    max_prob = 0
                    max_choice = None  # All 0s / uninitialized
                    for choice, prob in zip(self.choice_keys[key], out):
                        if prob.item() > max_prob:
                            max_prob = prob.item()
                            max_choice = choice
                    out = max_choice
            else:
                out = self.data[self.data_indices[key]]
                if self.is_embedding[key]:
                    out = get_name(int(out.item()))
                if self.is_int[key]:
                    out = int(out.item())
            return out
        else:
            self.throw_index_err(key)

    def __setattr__(self, key, value):
        if key in builtins:  # all keys that we need to operate on regularly
            return super().__setattr__(key, value)
        elif self.is_data(key):
            if key in self.slice_keys:
                if self.is_embedding[key]:
                    value = [get_idx(o) for o in value]
                if self.is_choice[key] and type(value) is str:
                    value = [choice == value for choice in self.choice_keys[key]]
                self.data[self.slice_keys[key]] = torch.tensor(value)
            else:
                if self.is_embedding[key]:
                    value = get_idx(value)
                self.data[self.data_indices[key]] = value
        else:
            self.throw_index_err(key)

    def __getitem__(self, key):
        if key in self.data_keys:
            return self.data[self.data_keys[key]]
        elif key in self.slice_keys:
            return self.data[self.slice_keys[key]]
        else:
            self.throw_index_err(key)

    def __setitem__(self, key, value):
        if key in self.data_keys:
            self.data[self.data_keys[key]] = value
        elif key in self.slice_keys:
            self.data[self.slice_keys[key]] = value
        else:
            self.throw_index_err(key)

    def throw_index_err(self, key):
        raise Exception(
            f"ERROR, {key} not found in {type(self)} object."
            f"Viable keys are {self.keys()}"
        )


if __name__ == "__main__":
    cfg = DataConfig("src/data/play.yaml")

    t = ThoughtPath(cfg)
