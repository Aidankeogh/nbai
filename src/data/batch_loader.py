from pbpstats.data_loader import StatsNbaPossessionLoader, StatsNbaGameFinderLoader
from src.utilities.global_timers import timeit, timers
from src.data.data_utils import team_name, get_prev
from src.data.box_stats import box_stats, parse_box_stats
from src.data.play import Play, parse_play, split_events
from src.data.possession import Possession, parse_possession
from src.data.game import Game, parse_game
from collections import defaultdict
import os
import numpy as np
import torch

@timeit
def parse_season(in_data, out_data):
    season = {}
    season['year'] = in_data['season'][0]
    season['type'] = in_data['season'][1]
    season['start_idx'] = get_prev(out_data['seasons'], 'end_idx')
    season['end_idx'] = len(out_data['plays'])
    season['start_game_idx'] = get_prev(out_data['seasons'], 'end_game_idx')
    season['end_game_idx'] = len(out_data['games'])
    season['start_pos_idx'] = get_prev(out_data['seasons'], 'end_pos_idx')
    season['end_pos_idx'] = len(out_data['possessions'])
    season['stats'] = {}
    season['rosters'] = defaultdict(set)
    plays = [Play(play) for play in out_data['plays'][season['start_idx']:season['end_idx']]]

    for play in plays:
        season['rosters'][play.offense_team].update(play.offense_roster)
        season['rosters'][play.defense_team].update(play.defense_roster)

    for key in season['rosters'].keys():
        season['rosters'][key] = list(season['rosters'][key])
        season['stats'][key] = box_stats(season['rosters'][key])
    season['rosters'] = dict(season['rosters'])
    
    for play in plays:
        try:
            off_stats, def_stats = parse_box_stats(play, season['rosters'])
            season['stats'][play.offense_team] += off_stats
            season['stats'][play.defense_team] += def_stats
        except Exception as e:
            print(e)

    out_data['seasons'].append(season)

def dump_season(out_data, db):
    season = out_data['seasons'][-1]
    play_key = "/".join([season['year'], season['type'], 'plays'])
    db[play_key] = torch.stack(out_data['plays'])
    possession_key = "/".join([season['year'], season['type'], 'possessions'])
    db[possession_key] = torch.stack(out_data['possessions'])
    possession_key = "/".join([season['year'], season['type'], 'games'])
    db[possession_key] = torch.stack(out_data['games'])

@timeit
def load_stats(config, db, years=range(2001,2020)):

    out_data = {
        'seasons': [],
        'games': [],
        'possessions': [],
        'plays': [],
    }

    in_data = {
        'season': None,
        'game': None,
        'possession': None,
        'play': None
    }

    season_iter = []
    for season_type in ["Playoffs", "Regular Season"]:
        season_iter.extend([(str(y-1) + "-" + str(y)[-2:], season_type) for y in years])

    for in_season in season_iter:
        game_iter = StatsNbaGameFinderLoader("nba", in_season[0], in_season[1], "file", "cache/data").items
        in_data['season'] = in_season
        for in_game in game_iter:
            possession_iter = StatsNbaPossessionLoader(in_game.data['game_id'], "file", "cache/data").items
            in_data['game'] = in_game
            for in_possession in possession_iter:
                in_data['possession'] = in_possession
                play_iter = split_events(in_possession.events)
                for in_play in play_iter:
                    in_data['play'] = in_play
                    parse_play(in_data, out_data)
                parse_possession(in_data, out_data)
            parse_game(in_data, out_data)
        parse_season(in_data, out_data)
    dump_season(out_data, db)
    return out_data

if __name__ == "__main__":
    from src.h5_db import get_connection 
    db = get_connection({}, name="test2")
    out_data = load_stats(None, db, years=[2018])
    box_stats = out_data['seasons'][0]['stats']['GSW']
    print(box_stats)
    steph_stats = box_stats['stephen-curry',:]
    incorrect_steph_stats = [379, 155, 73, 64, 162, 41, 38, 9, 82, 81, 26, 11, 43, 34, 1300, -1151, 1228, 1247]
    # This is not quite correct, check out https://www.basketball-reference.com/teams/GSW/2018.html#playoffs_totals
    print(steph_stats)
    print(list(steph_stats))
    assert list(steph_stats) == incorrect_steph_stats