from src.data.play import Play
from collections import defaultdict
import torch
import copy
from tabulate import tabulate


categories = ['pts', '2pa', '2pm', '3pm', '3pa', 'ftm',
              'fta', 'orb', 'drb', 'ast', 'stl', 'blk',
              'tov', 'pfs', 'o+-', 'd+-', 'o_pos', 'd_pos']


c = {categories[i] : i for i in range(len(categories))}
class Stats(defaultdict):
    def __init__(self, p, rosters):
        super().__init__(lambda: torch.zeros(len(categories)))
        
        if type(p) is Play:
            
            self.player_dict[p.offense_team] = set(p.off_team)
            self.player_dict[p.offense_team] = set(p.def_team)

            for player in p.offense_roster:
                self[player][c['o_pos']] = 1 - p.is_second_chance
                self[player][c['o+-']] = p.score_change
    
            for player in p.def_team:
                self[player][c['d_pos']] = 1 - p.is_second_chance
                self[player][c['d+-']] = -p.score_change

            if p.o_fga:
                self[p.o_fga][c['2pa']] = p.is2
                self[p.o_fga][c['2pm']] = p.make and p.is2
                self[p.o_fga][c['3pa']] = p.is3
                self[p.o_fga][c['3pm']] = p.make and p.is3
                self[p.o_fga][c['pts']] += p.make * (2 + p.is3)

            if p.o_ft:
                self[p.o_ft][c['fta']] = p.ft_made + p.ft_missed
                self[p.o_ft][c['ftm']] = p.ft_made
                self[p.o_ft][c['pts']] += p.ft_made

            if p.o_oreb:
                self[p.o_oreb][c['orb']] = 1

            if p.d_dreb:
                self[p.d_dreb][c['drb']] = 1

            if p.o_assist:
                self[p.o_assist][c['ast']] = 1
            if p.d_steal:
                self[p.d_steal][c['stl']] = 1
            if p.d_block:
                self[p.d_block][c['blk']] = 1
            if p.o_tov:
                self[p.o_tov][c['tov']] = 1

            if p.d_foul_over_limit:
                self[p.d_foul_over_limit][c['pfs']] = 1
            if p.d_foul_shot:
                self[p.d_foul_shot][c['pfs']] = 1

    def __add__(self, other): 
        out = copy.deepcopy(self)
        out.player_dict = self.player_dict
        for k, v in other.items():
            out[k] += v
        for k, v in other.player_dict.items():
            out.player_dict[k].update(v)
        return out
    
    def __repr__(self):
        table = []
        for team, players in self.player_dict.items():
            if team is None:
                team = "NONE"
            table.append(['--- ' + team + ' ---'] + ['-' for c in categories])
            for player in sorted(players):
                table.append([player] + list(self[player]))
        return tabulate(table, headers=[' '] + categories)


if __name__ == "__main__":
    p = Play()

    curry = player_id("stephen-curry")
    lebron = player_id("lebron-james")
    kd = player_id("kevin-durant")
    p.fga = 1
    p.is2 = 1
    p.make = 1
    p.foul_shot = 1
    p.ft_made = 1
    p.ft_missed = 0
    p.assist = 1
    p.o_assist = curry
    p.o_fga = kd
    p.o_ft = kd
    p.d_foul_shot = lebron
    p.off_team_name = "GSW"
    p.def_team_name = "CLE"
    p.off_team = [curry, curry, kd, kd, kd]
    p.def_team = [lebron, lebron, lebron, lebron, lebron]

    print(Stats(p))