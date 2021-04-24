import oyaml
import src.pipeline_utilities.load_configs

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

if __name__ == "__main__":
    (
        data_keys,
        data_indices,
        is_embedding,
        slice_keys,
        choice_keys,
        choice_indices
    ) = parse_yaml("src/play_path.yaml")