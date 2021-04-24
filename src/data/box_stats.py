
from src.data.data_utils import team_name
from src.embedding_utilities import get_name, get_idx
from src.thought_path import parse_yaml
from tabulate import tabulate
import torch

categories = ['pts', '2pa', '2pm', '3pm', '3pa', 'ftm',
              'fta', 'orb', 'drb', 'ast', 'stl', 'blk',
              'tov', 'pfs', 'o+-', 'd+-', 'o_pos', 'd_pos']
category_idxes = {category: i for i, category in enumerate(categories)}

class box_stats:
    def __init__(self, roster, data=None) -> None:
        if "none" in roster:
            print(roster)
            raise Exception("ERROR")
        self.roster = roster
        self.player_idxes = {player: i for i, player in enumerate(self.roster)}
        if data is None:
            self.data = torch.zeros(len(roster), len(categories))
        else:
            self.data = data

    def keys(self):
        return categories

    def __getitem__(self, key):
        player, category = key
        if player == "none":
            return None
        if type(player) is slice:
            player_idx = player
        else:
            player_idx = self.player_idxes[player]
        
        if type(category) is slice:
            category_idx = category
        else:
            category_idx = category_idxes[category]
        return self.data[player_idx, category_idx]


    def __setitem__(self, key, value):
        player, category = key
        if player != "none":
            if player not in self.player_idxes:
                print(player, self.player_idxes, value)
            player_idx = self.player_idxes[player]
            category_idx = category_idxes[category]
            self.data[player_idx, category_idx] = value

    def __add__(self, other):
        return box_stats(self.roster, self.data + other.data)

    def __iadd__(self, other):
        self.data += other.data
        return self
    
    def __repr__(self):
        table = []
        for player in sorted(self.roster):
            if player != "none":
                table.append([player] + list(self[player, :]))
        return tabulate(table, headers=[' '] + categories)