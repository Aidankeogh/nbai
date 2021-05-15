import yaml
import msgpack
import os

def get_embedding_indices():
    with open("metadata/team_ids.yaml", "r") as f:
        team_ids = yaml.safe_load(f)
    with open("metadata/player_info.yaml", "r") as f:
        player_info = yaml.safe_load(f)

    teams = []
    for k, v in team_ids.items():
        if type(k) is str:
            teams.append(k)

    player_sort_keys = []
    players = []
    for k, v in player_info.items():
        players.append(v["PLAYER_SLUG"])
        player_sort_keys.append(str(v['FROM_YEAR']) + "-" + v["BIRTHDATE"])
        
    zipped = zip(players, player_sort_keys)

    sorted_players = sorted(zipped, key = lambda x: x[1])
    sorted_players = [s[0] for s in sorted_players] 
    sorted_teams = sorted(teams)

    idx_to_name = []
    name_to_idx = {}
    for idx, name in enumerate(["none"] + sorted_teams + sorted_players):
        idx_to_name.append(name)
        name_to_idx[name] = idx

    with open("metadata/name_to_idx.msgpack", "wb") as f:
        msgpack.dump(name_to_idx, f) 
    with open("metadata/idx_to_name.msgpack", "wb") as f:
        msgpack.dump(idx_to_name, f) 

if not os.path.exists("metadata/name_to_idx.msgpack"):
    get_embedding_indices()

with open("metadata/name_to_idx.msgpack", "rb") as f:
    name_to_idx = msgpack.load(f) 
with open("metadata/idx_to_name.msgpack", "rb") as f:
    idx_to_name = msgpack.load(f) 

n_players = len(idx_to_name)

def get_name(idx):
    name = idx_to_name[idx]
    if name == "none":
        name = None
    return name

def get_idx(name):
    if name is None:
        name = "none"
    return name_to_idx[name]