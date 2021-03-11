from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader
from src.data_utils import player_name, player_id, team_name, team_id
from collections import defaultdict
from human_id import generate_id
import torch
import copy
import pickle

team_fields = [
    "off_team_name",
    "def_team_name"
]

player_fields = [
    # Players
    "o1",
    "o2",
    "o3",
    "o4",
    "o5",

    "d1",
    "d2",
    "d3",
    "d4",
    "d5",

    # Players who did things
    "o_fga",
    "o_tov",
    "o_foul_over_limit",
    "d_foul_over_limit",
    "o_foul_shot",
    "d_foul_shot",
    "o_assist",
    "d_block",
    "o_ft",
    "o_oreb",
    "d_dreb",
    "o_foul_off",
    "d_foul_off",
    "d_steal"
]

stat_fields = [
    # meta
    "is_second_chance",
    
    # Start of tree
    "fga",
    "tov",
    "foul_over_limit",

    # Off of fga
    "is2",
    "is3",
    "shot_dist",

    "foul_shot",

    "make",
    "miss",
    
    # Off of made shot
    "assist",

    # Off of missed shot 
    "block",
    
    # Off of foul events
    "ft_made",
    "ft_missed",

    # Off of miss or ft_miss
    "oreb",
    "dreb",
    
    # TOV types
    "foul_off",
    "steal",
    "other",
]

fields = team_fields + player_fields + stat_fields

teams = {
    "off_team": torch.arange(5) + len(team_fields),
    "def_team": torch.arange(5) + len(team_fields) + 5,
}

tensor_enum = {fields[i]: i for i in range(len(fields))}

def safe_id(key, value):
    if type(value) is str:
        if key in player_fields or key in teams:
            value = player_id(value)
        if key in team_fields:
            value = team_id(value)
    return value

class Play():
    def __init__(self, data=None, data_id=None):
        if data is None:
            self.id = "play-" + generate_id(word_count=6)
            self.data = torch.zeros(len(fields))
        else:
            self.data = data
            if data_id:
                self.id = data_id

    def __getattr__(self, key):
        if key in tensor_enum:
            out = self.data[tensor_enum[key]]
            if key in player_fields:
                out = player_name(out)
            if key in team_fields:
                out = team_name(out)
            return out
        elif key in teams:
            return [player_name(data) for data in self.data[teams[key]]]
        elif key == "score_change":
            return self.make * (2 + self.is3) + self.ft_made
        else:
            raise Exception(f"ERROR, attribute {key} not found")

    def __setattr__(self, key, value):
        if key in tensor_enum:
            self.data[tensor_enum[key]] = safe_id(key, value)
        elif key in teams:
            if type(value) is list:
                value = [safe_id(key, v) for v in value]
                value = torch.tensor(value, dtype=torch.float)
            self.data[teams[key]] = value
        else:
            return super().__setattr__(key, value)

if __name__ == "__main__":
    p = Play()

    p.fga = 1
    curry = player_id("stephen-curry")
    lebron = player_id("lebron-james")
    kd = player_id("kevin-durant")
    p.off_team = [curry, curry, kd, kd, kd]
    p.def_team = [lebron, lebron, lebron, lebron, lebron]
    assert(p.off_team[1] == "stephen-curry")
    assert(p.fga == 1)

