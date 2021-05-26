from pbpstats.data_loader.nba_possession_loader import NbaPossessionLoader
from src.utilities.global_timers import timeit
from tabulate import tabulate
import torch

categories = ['pts', '2pa', '2pm', '3pm', '3pa', 'ftm',
              'fta', 'orb', 'drb', 'ast', 'stl', 'blk',
              'tov', 'pfs', 'o+-', 'd+-', 'o_pos', 'd_pos']
category_idxes = {category: i for i, category in enumerate(categories)}

class Box_stats:
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
        if player is not None:
            #if player not in self.player_idxes:
                #print(player, self.player_idxes, value)
            player_idx = self.player_idxes[player]
            category_idx = category_idxes[category]
            self.data[player_idx, category_idx] = value

    def __add__(self, other):
        return Box_stats(self.roster, self.data + other.data)

    def __iadd__(self, other):
        self.data += other.data
        return self
    
    def __repr__(self):
        table = []
        for player in sorted(self.roster):
            if player is not None:
                table.append([player] + list(self[player, :]))
        return tabulate(table, headers=[' '] + categories)

@timeit
def parse_box_stats(play, rosters):
    offense_roster = rosters[play.offense_team]
    defense_roster = rosters[play.defense_team]

    o_stats = Box_stats(offense_roster)
    d_stats = Box_stats(defense_roster)

    for player in play.offense_roster:
        o_stats[player, 'o_pos'] = 1 - play.is_second_chance
        o_stats[player, 'o+-'] = play.score_change

    for player in play.defense_roster:
        d_stats[player, 'd_pos'] = 1 - play.is_second_chance
        d_stats[player, 'd+-'] = -play.score_change

    if play.shooter is not None:
        shot_type = max(play.shot_type, key=lambda key: play.shot_type[key])
        two_pointer = (shot_type == 'LongMidRange' or shot_type == 'AtRim' or shot_type == 'ShortMidRange')
        three_pointer = (shot_type == 'Arc3' or  shot_type == 'Corner3')

        o_stats[play.shooter, '2pa'] = two_pointer
        o_stats[play.shooter, '2pm'] = play.shot_made and not play.is_3
        o_stats[play.shooter, '3pa'] = three_pointer  
        o_stats[play.shooter, '3pm'] = play.shot_made and play.is_3
        o_stats[play.shooter, 'pts'] += play.shot_made * (2 + play.is_3)

    if play.free_thrower is not None and play.free_thrower in o_stats.player_idxes:
        #o_stats[play.o_ft, 'fta'] = play.ft_made + play.ft_missed
        n_free_throws = (play.initial_event["foul_over_limit"] == 1.0) * 2

        if play.shot_made and play.shot_fouled:
            n_free_throws = 1
        elif play.shot_fouled:
            n_free_throws = 2 + play.is_3
        free_throws_made = (
            play.first_free_throw_made + 
            play.middle_free_throw_made +
            play.last_free_throw_made
        )

        if free_throws_made > n_free_throws:
            n_free_throws = free_throws_made

        o_stats[play.free_thrower, 'ftm'] = free_throws_made
        o_stats[play.free_thrower, 'fta'] = n_free_throws
        o_stats[play.free_thrower, 'pts'] += free_throws_made
        
        # sanity check 
        if free_throws_made > n_free_throws:
            print("freethrows made:",free_throws_made)
            print("freethows attmpeted:",n_free_throws)
            print(play)

    if play.defensive_rebounder is not None:
        d_stats[play.defensive_rebounder, 'drb'] = 1

    if play.offensive_rebounder is not None:
        o_stats[play.offensive_rebounder, 'orb'] = 1

    if play.assister is not None:
        o_stats[play.assister, 'ast'] = 1
    if play.stealer is not None:
        d_stats[play.stealer, 'stl'] = 1
    if play.blocker is not None:
        d_stats[play.blocker, 'blk'] = 1
    if play.turnoverer is not None:
        o_stats[play.turnoverer, 'tov'] = 1

    if play.over_limit_fouler is not None and play.over_limit_fouler in d_stats.player_idxes:
        d_stats[play.over_limit_fouler, 'pfs'] = 1
    if play.shooting_fouler is not None and play.shooting_fouler in d_stats.player_idxes:
        d_stats[play.shooting_fouler, 'pfs'] = 1
    if play.common_fouler is not None and play.common_fouler in d_stats.player_idxes:
        d_stats[play.common_fouler, 'pfs'] = 1
    
    return o_stats, d_stats